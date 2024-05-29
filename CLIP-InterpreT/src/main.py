import socket
import gradio as gradio

from util import (
    change_model_layer, change_model_layer_head,
    change_model_property
)

from topic_seg_labels import (
    get_ViT_B_16_laion2b_s34b_b88k_layer8_head_anno
) 

from util_functions import (
    get_seg_mask, get_topic_mask,
    get_nearest_neighbours_image, get_nearest_neighbours_text,
    get_nearest_neighbours_image_with_properties
)

with gradio.Blocks() as demo:
    gradio.Markdown(
        """
        # CLIP-InterpreT
        CLIP-InterpreT is an interactive application for visualizing the contributions of specific heads in CLIP-like models via image representation decomposition.
        """
    )
    gradio.Markdown(
        """
        ### Start by selecting a model and uploading an image to analyze
        """
    )
    with gradio.Row():
        with gradio.Column(): 
            model_dropdown_choice = gradio.Dropdown(choices=['ViT-B-16_laion2b_s34b_b88k',
                                    'ViT-B-16_openai',
                                    'ViT-B-32_datacomp_m_s128m_b4k',
                                    'ViT-B-32_openai',
                                    'ViT-L-14_laion2b_s32b_b82k',
                                    'ViT-L-14_openai'], label="Model")
            image_input = gradio.Image(type="pil") 
        with gradio.Column(): 
            None
        with gradio.Column(): 
            None
    gradio.Markdown(
        """
        ### Select the tabs below for different decomposition analyses of your image
        """
    )
    with gradio.Tab("Property-based nearest neighbors search"):
        with gradio.Row():
            with gradio.Column(scale=1): 
                
                with gradio.Row():
                    property_dropdown_choice = gradio.Dropdown(choices=["animals", "locations", "art", "subject", "nature"], 
                                                            label="Property")
                    model_dropdown_choice.change(change_model_property, model_dropdown_choice, property_dropdown_choice)

                with gradio.Row():
                    btn = gradio.Button("Get nearest neighbors")
        
            with gradio.Column(scale=4): 
                with gradio.Row():
                    #plot = gradio.Plot()
                    plot = gradio.Image(label="Nearest neighbor images", type="filepath")

        btn.click(get_nearest_neighbours_image_with_properties,
                inputs=[image_input, model_dropdown_choice, property_dropdown_choice],
                outputs=[plot]
                )
    with gradio.Tab("Topic segmentation"):
        with gradio.Row():
            with gradio.Column(scale=1):
                with gradio.Row():
                    layer_dropdown_choice = gradio.Dropdown(choices=["8", "9", "10", "11"], label="Layer")
                    model_dropdown_choice.change(change_model_layer, model_dropdown_choice, layer_dropdown_choice)

                    head_dropdown_choice = gradio.Dropdown(choices=get_ViT_B_16_laion2b_s34b_b88k_layer8_head_anno(), 
                                                        label="Head")

                    layer_dropdown_choice.change(change_model_layer_head, [model_dropdown_choice, layer_dropdown_choice],
                                                head_dropdown_choice)
                    text_input = gradio.Textbox(label="Text input")       

                with gradio.Row():
                    btn = gradio.Button("Generate heatmap")

            with gradio.Column(scale=4):
                plot = gradio.Image(label="Heatmap", type="filepath")

        btn.click(get_topic_mask,
                inputs=[image_input, text_input, model_dropdown_choice, head_dropdown_choice],
                outputs=[plot]
                )
    with gradio.Tab("Contrastive segmentation"):
        with gradio.Row():
            with gradio.Column(scale=1):
                text_input1 = gradio.Textbox(label="First text")
                text_input2 = gradio.Textbox(label="Second text")

                
                with gradio.Row():
                    btn = gradio.Button("Generate  heatmap")

            with gradio.Column(scale=2):
                plot1 = gradio.Image(label="First text heatmap", type="filepath")
            with gradio.Column(scale=2):
                plot2 = gradio.Image(label="Second text heatmap", type="filepath")

        btn.click(get_seg_mask,
                inputs=[image_input, text_input1, text_input2, model_dropdown_choice],
                outputs=[plot1, plot2]
                )
    with gradio.Tab("Nearest Neighbours for an Image"):
        with gradio.Row():
                with gradio.Column(scale=1): 
                    with gradio.Row():
                        layer_dropdown_choice = gradio.Dropdown(choices=["8", "9", "10", "11"], label="Layer")
                        model_dropdown_choice.change(change_model_layer, model_dropdown_choice, layer_dropdown_choice)

                    with gradio.Row():
                        head_dropdown_choice = gradio.Dropdown(choices=get_ViT_B_16_laion2b_s34b_b88k_layer8_head_anno(), 
                                                            label="Head")

                        layer_dropdown_choice.change(change_model_layer_head, [model_dropdown_choice, layer_dropdown_choice],
                                                    head_dropdown_choice)

                    with gradio.Row():
                        btn = gradio.Button("Find nearest neighbours")

                with gradio.Column(scale=4):
                    with gradio.Row():
                        plot = gradio.Image(label="Nearest neighbour images", type="filepath")

        btn.click(get_nearest_neighbours_image,
                inputs=[image_input, model_dropdown_choice, head_dropdown_choice],
                outputs=[plot]
                )
    with gradio.Tab("Nearest Neighbours of an image for text inputs"):
        with gradio.Row():
            with gradio.Column(scale=1):
                with gradio.Row():
                    layer_dropdown_choice = gradio.Dropdown(choices=["8", "9", "10", "11"], label="Layer")
                    model_dropdown_choice.change(change_model_layer, model_dropdown_choice, layer_dropdown_choice)
                with gradio.Row():
                    head_dropdown_choice = gradio.Dropdown(choices=get_ViT_B_16_laion2b_s34b_b88k_layer8_head_anno(), 
                                                        label="Head")

                    layer_dropdown_choice.change(change_model_layer_head, [model_dropdown_choice, layer_dropdown_choice],
                                                head_dropdown_choice)
                    text_input = gradio.Textbox(label="Text input")

                with gradio.Row():
                    btn = gradio.Button("Get nearest neighbours for text input")

            with gradio.Column(scale=4):
                with gradio.Row():
                    plot = gradio.Image(label="Nearest neighbour images for text input", type="filepath")

        btn.click(get_nearest_neighbours_text,
                inputs=[text_input, model_dropdown_choice, head_dropdown_choice],
                outputs=[plot]
                )

host = socket.getfqdn() if 'internal' not in socket.getfqdn() else '0.0.0.0'
demo.launch(server_name=host, server_port=7860, share=False)
