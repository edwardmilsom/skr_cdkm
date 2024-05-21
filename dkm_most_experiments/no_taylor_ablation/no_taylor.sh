for seed in 0 1 2 3; do
	python ../resdkm20.py --ncol_div=4 --init_lr=0.01 --n_ind_scale=32 --dof=0.001 --final_layer=BFC --likelihood=categorical --seed=$seed --bn_indnorm="global" --bn_indscale="global" --bn_tnorm="global" --bn_tscale="global" --dtype=float32 --device="cuda" --data_folder_path="../../data/" --dataset=CIFAR10 --n_epochs=1200 --kernel=normalizedgaussian --c_noisevar 0 --no_taylor_obj
done
