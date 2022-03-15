from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
  'celeba_encode': {
    'transforms': transforms_config.EncodeTransforms,
    'train_source_root': dataset_paths['celeba_train'],
    'train_target_root': dataset_paths['celeba_train'],
    'test_source_root': dataset_paths['celeba_test'],
    'test_target_root': dataset_paths['celeba_test'],
  },
	'ffhq_aging': {
		'transforms': transforms_config.AgingTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'overfit': {
		'transforms': transforms_config.OverfitTransforms,
		'train_source_root': dataset_paths['overfit'],
		'train_target_root': dataset_paths['overfit_target'],
		'test_source_root': dataset_paths['overfit'],
		'test_target_root': dataset_paths['overfit_target'],
	}
}
