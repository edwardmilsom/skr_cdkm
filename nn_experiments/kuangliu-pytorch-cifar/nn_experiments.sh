for seed in 0 1 2 3; do
	python main_sgd.py --seed=$seed
done

for seed in 0 1 2 3; do
	python main_adam.py --seed=$seed
done

for seed in 0 1 2 3; do
	python main_sgd_cifar100.py --seed=$seed
done

for seed in 0 1 2 3; do
	python main_adamw_cifar100.py --seed=$seed
done