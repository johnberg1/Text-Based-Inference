dataset_paths = {
	'celeba_train': '',
	'celeba_test': '',
	'cub_train': '/kuacc/users/aanees20/dataset/CUB_Birds/CUB_200_2011/lmdb304New/', #list of filenames
	'cub_test': '/kuacc/users/aanees20/dataset/CUB_Birds/CUB_200_2011/lmdbtest256/',
	'cub_train_names': '/kuacc/users/aanees20/dataset/CUB_Birds/birds/train/filenames.pickle',
	'cub_test_names': '/kuacc/users/aanees20/dataset/CUB_Birds/birds/test/filenames.pickle',
	'cub_images_folder': '/scratch/users/aanees20/dataset/CUB_Birds/CUB_200_2011/images/',
	'cub_text_folder': '/kuacc/users/aanees20/dataset/CUB_Birds/birds/text/text/',
	'celeba_train_sketch': '',
	'celeba_test_sketch': '',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': '',
}

model_paths = {
	'stylegan_ffhq': '/kuacc/users/aanees20/pixel2style2pixel/pretrained_models/stylegan2-ffhq-config-ff.pt',
	'stylegan_cub': '/kuacc/users/aanees20/StyleGAN2_rosinality/stylegan2-pytorch/checkpoint/350000.pt',
	'psp_cub': '/kuacc/users/aanees20/pspCUBExperiments/exp5/checkpoints/best_model.pt',
	'psp_ffhq': '/kuacc/users/aanees20/pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt',
	'ir_se50': '/kuacc/users/aanees20/pixel2style2pixel/pretrained_models/model_ir_se50.pth',
	'circular_face': '/kuacc/users/aanees20/pixel2style2pixel/pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': '/kuacc/users/aanees20/pixel2style2pixel/pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': '/kuacc/users/aanees20/pixel2style2pixel/pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': '/kuacc/users/aanees20/pixel2style2pixel/pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': '/kuacc/users/aanees20/pixel2style2pixel/pretrained_models/moco_v2_800ep_pretrain.pt'
}