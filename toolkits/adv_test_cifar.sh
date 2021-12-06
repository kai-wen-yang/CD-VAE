python tools/adv_test_cifar.py --model_path ./results/defense_0.1_0.1/robust_model_g_epoch82.pth --vae_path ./results/defense_0.1_0.1/robust_vae_epoch82.pth --batch_size 256 \
"NoAttack()" \
"AutoLinfAttack(cd_vae, 'cifar', bound=8/255)" \
"AutoL2Attack(cd_vae, 'cifar', bound=1.0)" \
"JPEGLinfAttack(cd_vae, 'cifar', bound=0.125, num_iterations=100)" \
"StAdvAttack(cd_vae, num_iterations=100)" \
"ReColorAdvAttack(cd_vae, num_iterations=100)"