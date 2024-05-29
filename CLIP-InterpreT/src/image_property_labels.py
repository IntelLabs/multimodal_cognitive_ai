def get_ViT_B_16_laion2b_s34b_b88k_property_head_layer_labels(property):
    property = property.lower()
    properties = {'animals': ['L9.H2', 'L10.H4'],
                'locations': ['L9.H8', 'L10.H3', 'L11.H0', 'L11.H6'],
                'art': ['L9.H10', 'L9.H11'],
                'subject': ['L8.H1', 'L8.H4', 'L8.H10', 'L8.H11'],
                'nature': ['L8.H7', 'L9.H3', 'L9.H7', 'L11.H2'],

                }

    return properties[property]

def get_ViT_B_16_openai_property_head_layer_labels(property):
    property = property.lower()
    properties = {'animals': ['L9.H10', 'L10.H5', 'L11.H1'],
                'locations': ['L8.H1', 'L8.H7', 'L8.H11', 'L9.H3', 'L9.H7', 'L10.H2',
                            'L10.H3', 'L10.H4', 'L10.H7', 'L10.H10', 'L11.H6'],
                }

    return properties[property]

def get_ViT_B_32_datacomp_m_s128m_b4k_property_head_layer_labels(property):
    property = property.lower()
    properties = {'animals': ['L8.H2', 'L8.H3', 'L11.H1', 'L11.H6'],
                'colors': ['L9.H6', 'L11.H4', 'L11.H9'],
                }

    return properties[property]

def get_ViT_B_32_openai_property_head_layer_labels(property):
    property = property.lower()
    properties = {'photography': ['L8.H6', 'L8.H7', 'L9.H6', 'L10.H1'],
                'pattern': ['L8.H2', 'L8.H9'],
                'locations': ['L8.H4', 'L8.H8', 'L9.H11', 'L10.H3', 'L10.H7', 'L10.H10',
                            'L11.H6', 'L11.H9', 'L11.H10', 'L11.H11']}

    return properties[property]

def get_ViT_L_14_laion2b_s32b_b82k_property_head_layer_labels(property):
    property = property.lower()
    properties = {'colors': ['L21.H0', 'L21.H9', 'L22.H10', 'L22.H11', 'L22.H14', 'L23.H8'],
                'locations': ['L20.H0', 'L20.H1', 'L20.H2', 'L20.H3', 'L20.H8', 'L20.H9',
                            'L21.H1', 'L21.H3', 'L21.H11', 'L21.H13', 'L21.H14',
                            'L22.H2', 'L22.H12', 'L22.H13', 'L23.H6'],
                'environment': ['L20.H6', 'L21.H6', 'L22.H7', 'L23.H5'],
                'objects': ['L20.H10', 'L21.H7', 'L22.H3', 'L23.H7'],
                'photography': ['L20.H11', 'L20.H13', 'L23.H13']}

    return properties[property]

def get_ViT_L_14_openai_property_head_layer_labels(property):
    property = property.lower()
    properties = {'colors': ['L21.H4'],
                'locations': ['L20.H0', 'L20.H2', 'L20.H3', 'L20.H10',
                            'L21.H1', 'L21.H10', 'L21.H11', 'L21.H13', 'L21.H15',
                            'L22.H2', 'L22.H5', 'L22.H14', 'L22.H15', 'L23.H0', 'L23.H6'],
                'environment': ['L20.H5', 'L22.H6', 'L23.H1', 'L23.H14'],
                'texture': ['L20.H13', 'L21.H2', 'L22.H7'],
                'wildlife': ['L20.H1', 'L20.H9', 'L22.H13'],
                'birds': ['L22.H13', 'L23.H7'],
                'clothing': ['L23.H4', 'L23.H13']}

    return properties[property]
