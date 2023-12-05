
from nice_slam.src import conv_onet


method_dict = {
    'conv_onet': conv_onet
}


# Models
def get_model(cfg, nice=True):
    """
    Returns the model instance.

    Args:
        cfg (dict): config dictionary.
        nice (bool, optional): if use NICE-SLAM. Defaults to True.

    Returns:
       model (nn.module): network model.
    """

    method = 'conv_onet'
    model = method_dict[method].config.get_model(
        cfg,  nice=nice)

    return model
