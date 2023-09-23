"""Loads a trained model checkpoint and makes predictions on a dataset."""

from .chemprop.parsing import parse_predict_args
from .chemprop.train.smiles_predictions import make_predictions, load_models

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Forward_IR:
    def __init__(self, checkpoint_dir : str, use_gpu : bool = False):
        checkpoint_paths = [
        checkpoint_dir+"/model_0.pt",
        checkpoint_dir+"/model_1.pt",	
        checkpoint_dir+"/model_2.pt",
        checkpoint_dir+"/model_3.pt",
        checkpoint_dir+"/model_4.pt",
        checkpoint_dir+"/model_5.pt",
        ]

        # TODO change for multiple GPUs 
        if use_gpu:
            use_cuda = True
            device = 'cuda:0'
        else:
            use_cuda = False
            device = 'cpu'

        self.args = Namespace(
        gpu=None,
        use_compound_names=False,
        preds_path='temp.csv',
        checkpoint_dir=checkpoint_dir,
        batch_size=50,
        features_generator=None,
        features_path=None,
        max_data_size=None,
        ensemble_variance=False,
        ensemble_variance_conv=0.0,
        checkpoint_paths=checkpoint_paths,
        ensemble_size=len(checkpoint_paths),
        cuda=use_cuda,
        device=device,
        )

        self.load_models()

    def load_models(self):
        self.models = load_models(self.args)
        print("Loaded all checkpoint models")
    
    def predict_smiles(self, smiles_string: str):
        preds = make_predictions(self.args, self.models, [smiles_string])
        return preds[0]


def get_IR_prediction(smiles_string: str, checkpoint_dir : str, checkpoint_paths : list[str], use_gpu : bool = False):
    checkpoint_paths = [
    checkpoint_dir+"/model_0.pt",
	checkpoint_dir+"/model_1.pt",	
	checkpoint_dir+"/model_2.pt",
	checkpoint_dir+"/model_3.pt",
	checkpoint_dir+"/model_4.pt",
	checkpoint_dir+"/model_5.pt",
	]

    # TODO change for multiple GPUs 
    if use_gpu:
        use_cuda = True
        device = 'cuda:0'
    else:
        use_cuda = False
        device = 'cpu'

    args = Namespace(
    gpu=None,
    use_compound_names=False,
    preds_path='temp.csv',
    checkpoint_dir=checkpoint_dir,
    batch_size=50,
    features_generator=None,
    features_path=None,
    max_data_size=None,
    ensemble_variance=False,
    ensemble_variance_conv=0.0,
    checkpoint_paths=checkpoint_paths,
    ensemble_size=len(checkpoint_paths),
    cuda=use_cuda,
    device=device,
    )
    preds = make_predictions(args, [smiles_string])
    #print(len(preds[0]),preds[0])
    return preds[0]
