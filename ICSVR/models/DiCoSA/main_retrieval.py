from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from os.path import join, exists

import torch
import torch.nn.functional as F
from models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.data_dataloaders import DATALOADER_DICT
from dataloaders.dataloader_msrvtt_retrieval import MSRVTTDataset
from models.modeling import DiCoSA, AllGather
from models.optimization import BertAdam
from utils.metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim

from utils.comm import is_main_process, synchronize
from utils.logger import setup_logger
from utils.metric_logger import MetricLogger

allgather = AllGather.apply

global logger


def get_args(description='Disentangled Representation Learning for Text-Video Retrieval'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", type=int, default=0, help="Whether to run training.")
    parser.add_argument("--do_eval", type=int, default=0, help="Whether to run evaluation.")
    parser.add_argument("--typ", type=str, default="straight", help="Type of data.")
    parser.add_argument("--testtyp", type=str, default="full")

    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument('--anno_path', type=str, default='data/MSR-VTT/anns', help='annotation path')
    parser.add_argument('--video_path', type=str, default='data/MSR-VTT/videos', help='video path')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--coef_lr', type=float, default=1e-3, help='coefficient for bert branch.')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument('--weight_decay', type=float, default=0.2, help='weight decay')
    parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=128, help='batch size eval')

    parser.add_argument('--max_words', type=int, default=32, help='max text token number')
    parser.add_argument('--max_frames', type=int, default=12, help='max key frames')
    parser.add_argument('--video_framerate', type=int, default=1, help='framerate to sample video frame')

    parser.add_argument("--device", default='cpu', type=str, help="cpu/cuda")
    parser.add_argument("--world_size", default=1, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--distributed", default=0, type=int, help="multi machine DDP")

    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--base_encoder", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument('--agg_module', type=str, default="seqTransf", choices=["None", "seqLSTM", "seqTransf"],
                        help="choice a feature aggregation module for video.")
    parser.add_argument('--interaction', type=str, default='wti', help="interaction type for retrieval.")
    parser.add_argument('--num_hidden_layers', type=int, default=4)
    
    parser.add_argument('--temp', type=float, default=5)

    parser.add_argument('--center', type=int, default=8)
    
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.005)
    
    parser.add_argument("--t2v_k", default=1, type=int)
    parser.add_argument("--t2v_beta", default=70, type=float)
    parser.add_argument("--t2v_theta", default=3, type=float)
    parser.add_argument("--v2t_k", default=1, type=int)
    parser.add_argument("--v2t_beta", default=70, type=float)
    parser.add_argument("--v2t_theta", default=5, type=float)
    
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")

    args = parser.parse_args()

    return args


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if torch.cuda.is_available():
        torch.distributed.barrier()
    logger.info("local_rank: {} world_size: {}".format(args.local_rank, args.world_size))

    if args.batch_size % args.world_size != 0 or args.batch_size_val % args.world_size != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and world_size parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.world_size, args.batch_size_val, args.world_size))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def build_model(args):
    model = DiCoSA(args)
    if args.init_model:
        if not exists(args.init_model):
            raise FileNotFoundError
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)

    model.to(args.device)
    return model


def build_dataloader(args):
    ## ####################################
    # dataloader loading
    ## ####################################
    tokenizer = ClipTokenizer()
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if isinstance(test_length, int):
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
    elif len(test_length) == 2:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %dt %dv", test_length[0], test_length[1])
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d %d", len(test_dataloader[0]), len(test_dataloader[1]))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %dt %dv", val_length[0], val_length[1])

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", len(train_dataloader) * args.epochs)
    else:
        train_dataloader, train_sampler = None, None
        
    train_dataloader0, train_length, train_sampler0 = DATALOADER_DICT[args.datatype]["train_test"](args, tokenizer)

    return test_dataloader, val_dataloader, train_dataloader, train_sampler, train_dataloader0, train_sampler0


def prep_optimizer(args, model, num_train_optimization_steps, local_rank):
    if hasattr(model, 'module'):
        model = model.module
    lr = args.lr  # 0.0001
    coef_lr = args.coef_lr  # 0.001
    weight_decay = args.weight_decay  # 0.2
    warmup_proportion = args.warmup_proportion
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    if torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                          find_unused_parameters=True)

    return optimizer, scheduler, model


def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file


def reduce_loss(loss, args):
    world_size = args.world_size
    if world_size < 2:
        return loss
    with torch.no_grad():
        torch.distributed.reduce(loss, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    return loss


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                scheduler, global_step, max_steps, val_dataloader):
    global logger
    global best_score
    global meters

    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    total_loss = 0
    
    end = time.time()
    logit_scale = 0
    for step, batch in enumerate(train_dataloader, start=1):
        global_step += 1
        data_time = time.time() - end

        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        text_ids, text_mask, video, video_mask, inds, idx = batch
        loss, uniformity_loss, alignment_loss = model(text_ids, text_mask, video, video_mask, idx, global_step)
        
        if n_gpu > 1:
            # print(loss.shape)
            loss = loss.mean()  # mean() to average on multi-gpu.
            uniformity_loss = uniformity_loss.mean()
            alignment_loss = alignment_loss.mean()

        with torch.autograd.detect_anomaly():
            loss.backward()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()  # Update learning rate schedule
        
        optimizer.zero_grad()

        # https://github.com/openai/CLIP/issues/46
        if hasattr(model, 'module'):
            torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.module.clip.logit_scale.exp().item()
        else:
            torch.clamp_(model.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.clip.logit_scale.exp().item()

        batch_time = time.time() - end
        end = time.time()

        reduced_l = reduce_loss(loss, args)
        reduced_uniformity_loss = reduce_loss(uniformity_loss, args)
        reduced_alignment_loss = reduce_loss(alignment_loss, args)
        meters.update(time=batch_time, data=data_time, loss=float(reduced_l),
                      E_loss=float(reduced_uniformity_loss), M_loss=float(reduced_alignment_loss))

        eta_seconds = meters.time.global_avg * (max_steps - global_step)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if (global_step % log_step == 0 or global_step == 1) and is_main_process():
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "epoch: {epoch}/{max_epoch}",
                        "iteration: {iteration}/{max_iteration}",
                        "{meters}",
                        "lr: {lr}",
                        "logit_scale: {logit_scale:.2f}"
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    epoch=epoch,
                    max_epoch=args.epochs,
                    iteration=global_step,
                    max_iteration=max_steps,
                    meters=str(meters),
                    lr="/".join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                    logit_scale=logit_scale,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if global_step % (log_step * 3) == 0 or global_step == 1:
            R1 = eval_epoch(args, model, val_dataloader, args.device)
            if args.local_rank == 0:
                output_model_file = save_model(epoch, args, model, type_name="step{}".format(global_step))
                if best_score <= R1:
                    best_score = R1
                    output_model_file = save_model(epoch, args, model, type_name="best")
            model.train()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(args, model, t_mask_list, v_mask_list, t_feat_list, v_feat_list, cls_list,
                       train_t_mask_list, train_v_mask_list, train_t_feat_list, train_v_feat_list,
                       train_cls_list, mini_batch=16):
    # Returns list of retrieved top k videos based on the sims matrix
    def get_retrieved_videos(sims, k, theta):
        argm = np.argsort(-sims, axis=1)
        topk = argm[:,:k].reshape(-1)
        retrieved_videos, occurrence_count = np.unique(topk, return_counts=True)
        return retrieved_videos[occurrence_count>=theta]

    # Returns list of indices to normalize from sims based on videos
    def get_index_to_normalize(sims, videos):
        argm = np.argsort(-sims, axis=1)[:,0]
        result = np.array(list(map(lambda x: x in videos, argm)))
        result = np.nonzero(result)
        return result

    def qb_norm(train_test, test_test, k, beta, theta):
        retrieved_videos = get_retrieved_videos(train_test, k, theta)
        test_test_normalized = test_test
        train_test = np.exp(train_test*beta)
        test_test = np.exp(test_test*beta)

        normalizing_sum = np.sum(train_test, axis=0)
        index_for_normalizing = get_index_to_normalize(test_test, retrieved_videos)
        test_test_normalized[index_for_normalizing, :] = \
            np.divide(test_test[index_for_normalizing, :], normalizing_sum)
        return test_test_normalized
    
    sim_matrix = []
    logger.info('[finish] map to main gpu')

    batch_t_mask = torch.split(t_mask_list, mini_batch)
    batch_v_mask = torch.split(v_mask_list, mini_batch)
    batch_t_feat = torch.split(t_feat_list, mini_batch)
    batch_v_feat = torch.split(v_feat_list, mini_batch)
    batch_cls_feat = torch.split(cls_list, mini_batch)
    
    train_batch_t_mask = torch.split(train_t_mask_list, mini_batch)
    train_batch_v_mask = torch.split(train_v_mask_list, mini_batch)
    train_batch_t_feat = torch.split(train_t_feat_list, mini_batch)
    train_batch_v_feat = torch.split(train_v_feat_list, mini_batch)
    train_batch_cls_feat = torch.split(train_cls_list, mini_batch)

    logger.info('[finish] map to main gpu')
    with torch.no_grad():        
        for idx1, (t_mask, t_feat, cls) in enumerate(zip(batch_t_mask, batch_t_feat, batch_cls_feat)):
            each_row = []
            for idx2, (v_mask, v_feat) in enumerate(zip(batch_v_mask, batch_v_feat)):
                logits, *_tmp = model.get_similarity_logits(t_feat, cls, v_feat, t_mask, v_mask)
                each_row.append(logits.cpu().detach().numpy())
            each_row = np.concatenate(tuple(each_row), axis=-1)
            sim_matrix.append(each_row)
            
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    
    train_test_t2v, train_test_v2t = [], []
    
    with torch.no_grad():
        for idx1, (t_mask, t_feat, cls) in enumerate(zip(train_batch_t_mask, train_batch_t_feat, train_batch_cls_feat)):
            each_row = []
            for idx2, (v_mask, v_feat) in enumerate(zip(batch_v_mask, batch_v_feat)):
                logits, _, *_tmp = model.get_similarity_logits(t_feat, cls, v_feat, t_mask, v_mask)
                logits = logits.cpu().detach().numpy()
                each_row.append(logits)
            each_row = np.concatenate(tuple(each_row), axis=-1)
            train_test_t2v.append(each_row)
    train_test_t2v = np.concatenate(tuple(train_test_t2v), axis=0)
    
    with torch.no_grad():
        for idx1, (t_mask, t_feat, cls) in enumerate(zip(batch_t_mask, batch_t_feat, batch_cls_feat)):
            each_row = []
            for idx2, (v_mask, v_feat) in enumerate(zip(train_batch_v_mask, train_batch_v_feat)):
                logits, _, *_tmp = model.get_similarity_logits(t_feat, cls, v_feat, t_mask, v_mask)
                logits = logits.cpu().detach().numpy()
                each_row.append(logits)
            each_row = np.concatenate(tuple(each_row), axis=-1)
            train_test_v2t.append(each_row)
    train_test_v2t = np.concatenate(tuple(train_test_v2t), axis=0)
    
    t2v_normalized = qb_norm(train_test_t2v.copy(), sim_matrix.copy(), args.t2v_k, args.t2v_beta, args.t2v_theta)
    v2t_normalized = qb_norm(train_test_v2t.T.copy(), sim_matrix.T.copy(), args.v2t_k, args.v2t_beta, args.v2t_theta)
    
    return sim_matrix, t2v_normalized, v2t_normalized


def eval_epoch(args, model, test_dataloader, device):
    global train_dataloader0
    global train_sampler0
    
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]
        
    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    # ----------------------------
    # 1. cache the features
    # ----------------------------
    batch_mask_t, batch_mask_v, batch_feat_t, batch_feat_v, ids_t, ids_v = [], [], [], [], [], []
    batch_cls = []
    tran_cls, train_t, train_v, train_mask_t, train_mask_v = [], [], [], [], []

    batch_mask_train_t, batch_mask_train_v, batch_feat_train_t, batch_feat_train_v, ids_train_t, ids_train_v = [], [], [], [], [], []
    batch_train_cls = []
    
    with torch.no_grad():
        logger.info('[start] extract train feature')
        for bid, batch in enumerate(train_dataloader0):
            batch = tuple(t.to(device) for t in batch)
            text_ids, text_mask, video, video_mask, inds, _ = batch
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            text_feat, video_feat, cls = model.get_text_video_feat(text_ids, text_mask, video, video_mask)
            ids_train_t.append(inds)
            batch_mask_train_t.append(text_mask)
            batch_mask_train_v.append(video_mask)
            batch_feat_train_t.append(text_feat)
            batch_feat_train_v.append(video_feat)
            batch_train_cls.append(cls)
            print("{}/{}\r".format(bid, len(train_dataloader0)), end="")

        ids_train_t = allgather(torch.cat(ids_train_t, dim=0), args).squeeze()
        batch_mask_train_t = allgather(torch.cat(batch_mask_train_t, dim=0), args)
        batch_mask_train_v = allgather(torch.cat(batch_mask_train_v, dim=0), args)
        batch_feat_train_t = allgather(torch.cat(batch_feat_train_t, dim=0), args)
        batch_feat_train_v = allgather(torch.cat(batch_feat_train_v, dim=0), args)
        batch_train_cls = allgather(torch.cat(batch_train_cls, dim=0), args)
        logger.info('[finish] extract train feature')
                
        tic = time.time()
        if multi_sentence_:  # multi-sentences retrieval means: one clip has two or more descriptions.
            total_video_num = 0
            logger.info('[start] extract text+video feature')
            for batch in tqdm(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                text_ids, text_mask, video, video_mask, inds, _ = batch

                b, *_t = video.shape
                text_feat, cls = model.get_text_feat(text_ids, text_mask)
                ids_t.append(inds)
                batch_mask_t.append(text_mask)
                batch_feat_t.append(text_feat)
                batch_cls.append(cls)

                video_feat = model.get_video_feat(video, video_mask)
                batch_mask_v.append(video_mask)
                batch_feat_v.append(video_feat)

                total_video_num += b

            ids_t = torch.cat(ids_t, dim=0).squeeze()
            batch_mask_t = torch.cat(batch_mask_t, dim=0)
            batch_mask_v = torch.cat(batch_mask_v, dim=0)
            batch_feat_t = torch.cat(batch_feat_t, dim=0)
            batch_feat_v = torch.cat(batch_feat_v, dim=0)
            batch_cls = torch.cat(batch_cls, dim=0)

            _batch_feat_v, _batch_mask_v = [], []
            for i in range(len(ids_t)):
                if ids_t[i] in cut_off_points_:
                    _batch_feat_v.append(batch_feat_v[i])
                    _batch_mask_v.append(batch_mask_v[i])

            batch_feat_v = torch.stack(_batch_feat_v, dim=0)
            batch_mask_v = torch.stack(_batch_mask_v, dim=0)

            logger.info('[finish] extract text+video feature')
        else:
            logger.info('[start] extract text+video feature')
            for batch in tqdm(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                text_ids, text_mask, video, video_mask, inds, _ = batch
                text_feat, video_feat, cls = model.get_text_video_feat(text_ids, text_mask, video, video_mask)
                ids_t.append(inds)
                batch_mask_t.append(text_mask)
                batch_mask_v.append(video_mask)
                batch_feat_t.append(text_feat)
                batch_feat_v.append(video_feat)
                batch_cls.append(cls)
            ids_t = allgather(torch.cat(ids_t, dim=0), args).squeeze()
            batch_mask_t = allgather(torch.cat(batch_mask_t, dim=0), args)
            batch_mask_v = allgather(torch.cat(batch_mask_v, dim=0), args)
            batch_feat_t = allgather(torch.cat(batch_feat_t, dim=0), args)
            batch_feat_v = allgather(torch.cat(batch_feat_v, dim=0), args)
            batch_cls = allgather(torch.cat(batch_cls, dim=0), args)
            batch_mask_t[ids_t] = batch_mask_t.clone()
            batch_mask_v[ids_t] = batch_mask_v.clone()
            batch_feat_t[ids_t] = batch_feat_t.clone()
            batch_feat_v[ids_t] = batch_feat_v.clone()
            batch_cls[ids_t] = batch_cls.clone()
            batch_mask_t = batch_mask_t[:ids_t.max() + 1, ...]
            batch_mask_v = batch_mask_v[:ids_t.max() + 1, ...]
            batch_feat_t = batch_feat_t[:ids_t.max() + 1, ...]
            batch_feat_v = batch_feat_v[:ids_t.max() + 1, ...]
            batch_cls = batch_cls[:ids_t.max() + 1, ...]
            logger.info('[finish] extract text+video feature')

    toc1 = time.time()

    logger.info('{} {} {} {}'.format(len(batch_mask_t), len(batch_mask_v), len(batch_feat_t), len(batch_feat_v)))
    # ----------------------------------
    # 2. calculate the similarity
    # ----------------------------------
    logger.info('[start] calculate the similarity')
    with torch.no_grad():
        sim_matrix0, t2v_sim_matrix, v2t_sim_matrix = _run_on_single_gpu(args, model, batch_mask_t, batch_mask_v, batch_feat_t, batch_feat_v, batch_cls, batch_mask_train_t, batch_mask_train_v, batch_feat_train_t, batch_feat_train_v, batch_train_cls)
    logger.info('[end] calculate the similarity')

    toc2 = time.time()
    logger.info('[start] compute_metrics')
    if multi_sentence_:  
        logger.info("before reshape, sim matrix size: {} x {}".format(t2v_sim_matrix.shape[0], t2v_sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
        t2v_sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            t2v_sim_matrix_new.append(np.concatenate((t2v_sim_matrix[s_:e_],
                                                  np.full((max_length - e_ + s_, t2v_sim_matrix.shape[1]), -np.inf)),
                                                 axis=0))
        t2v_sim_matrix_new = np.stack(tuple(t2v_sim_matrix_new), axis=0)

        v2t_sim_matrix = v2t_sim_matrix.T
        v2t_sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            v2t_sim_matrix_new.append(np.concatenate((v2t_sim_matrix[s_:e_],
                                                  np.full((max_length - e_ + s_, v2t_sim_matrix.shape[1]), -np.inf)),
                                                 axis=0))
        v2t_sim_matrix_new = np.stack(tuple(v2t_sim_matrix_new), axis=0)

        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(t2v_sim_matrix_new.shape[0], t2v_sim_matrix_new.shape[1], t2v_sim_matrix_new.shape[2]))

        tv_metrics = compute_metrics(t2v_sim_matrix_new)
        vt_metrics = compute_metrics(v2t_sim_matrix_new.T)
    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix0.shape[0], sim_matrix0.shape[1]))
        tv_metrics = compute_metrics(t2v_sim_matrix)
        vt_metrics = compute_metrics(v2t_sim_matrix)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix0), len(sim_matrix0[0])))

    logger.info('[end] compute_metrics')

    toc3 = time.time()
    logger.info("time profile: feat {:.1f}s match {:.5f}s metrics {:.5f}s".format(toc1 - tic, toc2 - toc1, toc3 - toc2))

    logger.info("Text-to-Video: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['R50'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['R50'], vt_metrics['MR'], vt_metrics['MeanR']))

    return tv_metrics['R1']


def main():
    global logger
    global best_score
    global meters
    global train_dataloader0
    global train_sampler0

    meters = MetricLogger(delimiter="  ")
    args = get_args()
    if not exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('tvr', args.output_dir, args.local_rank)

    args = set_seed_logger(args)

    model = build_model(args)

    test_dataloader, val_dataloader, train_dataloader, train_sampler, train_dataloader0, train_sampler0 = build_dataloader(args)
    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        tic = time.time()
        max_steps = len(train_dataloader) * args.epochs
        _max_steps = len(train_dataloader) * 5
        optimizer, scheduler, model = prep_optimizer(args, model, _max_steps, args.local_rank)

        best_score = 0.00001
        best_output_model_file = "None"
        global_step = 0
        for epoch in range(args.epochs):
            if train_sampler is not None: train_sampler.set_epoch(epoch)
            synchronize()
            torch.cuda.empty_cache()
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader,
                                               args.device, args.world_size, optimizer,
                                               scheduler, global_step, max_steps, val_dataloader)
            torch.cuda.empty_cache()
            R1 = eval_epoch(args, model, val_dataloader, args.device)
            torch.cuda.empty_cache()
            synchronize()

            if args.local_rank == 0:
                output_model_file = save_model(epoch, args, model, type_name="")

                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                    torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                               'best.pth')
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))
            synchronize()
        toc = time.time() - tic
        training_time = time.strftime("%Hh %Mmin %Ss", time.gmtime(toc))
        logger.info("*" * 20 + '\n' + f'training finished with {training_time}' + "*" * 20 + '\n')

        # test on the best checkpoint
        model = model.module
        if args.local_rank == 0:
            model.load_state_dict(torch.load('best.pth', map_location='cpu'), strict=False)
        if torch.cuda.is_available():
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              find_unused_parameters=True)

        torch.cuda.empty_cache()
        eval_epoch(args, model, test_dataloader, args.device)
        synchronize()

    elif args.do_eval:
        eval_epoch(args, model, test_dataloader, args.device)


if __name__ == "__main__":
    main()
