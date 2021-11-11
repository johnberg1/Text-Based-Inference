dataset_paths = {
    'celeba_train': '/scratch/users/abaykal20/sam/SAM/mmcelebhq/train_images',
    'celeba_test': '/scratch/users/abaykal20/sam/SAM/mmcelebhq/test_images',
    'ffhq': '',
    'overfit': '/scratch/users/abaykal20/sam/SAM/overfit_image',
    'overfit_target': '/scratch/users/abaykal20/sam/SAM/overfit_image_target'
}

model_paths = {
    'pretrained_psp_encoder': 'pretrained_models/psp_ffhq_encode.pt',
    'ir_se50': 'pretrained_models/model_ir_se50.pth',
    'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
    'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
    'age_predictor': 'pretrained_models/dex_age_classifier.pth'
}
