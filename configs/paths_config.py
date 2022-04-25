dataset_paths = {
    'celeba_train': '/scratch/users/abaykal20/sam/SAM/mmcelebhq/train_images',
    'celeba_test': '/scratch/users/abaykal20/sam/SAM/mmcelebhq/test_images',
    'ffhq': '',
    'overfit': '/scratch/users/abaykal20/sam/SAM/overfit_image_old',
    'overfit_target': '/scratch/users/abaykal20/sam/SAM/overfit_image_old'
}

model_paths = {
    'pretrained_psp_encoder': '/scratch/users/abaykal20/pSp/pixel2style2pixel/exp2/checkpoints/best_model.pt',
    'pretrained_e4e_encoder': 'pretrained_models/e4e_ffhq_encode.pt',
    'ir_se50': 'pretrained_models/model_ir_se50.pth',
    'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
    'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
    'age_predictor': 'pretrained_models/dex_age_classifier.pth'
}
