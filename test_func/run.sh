# Qudratic function
python main.py --n_iter 8000 \
    --lr_y 0.2 \
    --r 1 \
    --init_x 1 \
    --init_y 0.01 \
    --func quadratic \
    --L 2 \
    --ylim_top 1e4 \
    --ylim_bottom 3e-3 \
    --plot_sample 100 \
    --plot_size 3 \
    --plot_smooth 0.5 \
    --alpha 0.8 \
    --save \

python main.py --n_iter 8000 \
    --lr_y 0.2 \
    --r 2 \
    --init_x 1 \
    --init_y 0.01 \
    --func quadratic \
    --L 2 \
    --ylim_top 1e2 \
    --ylim_bottom 3e-3 \
    --plot_sample 100 \
    --plot_size 3 \
    --plot_smooth 0.6 \
    --alpha 0.8 \
    --save \

python main.py --n_iter 2000 \
    --lr_y 0.2 \
    --r 4 \
    --init_x 1 \
    --init_y 0.01 \
    --func quadratic \
    --L 2 \
    --ylim_top 1e1 \
    --ylim_bottom 1e-5 \
    --plot_sample 100 \
    --plot_size 3 \
    --plot_smooth 0.5 \
    --alpha 0.8 \
    --save \

python main.py --n_iter 5000 \
    --lr_y 0.2 \
    --r 8 \
    --init_x 1 \
    --init_y 0.01 \
    --func quadratic \
    --L 2 \
    --ylim_top 1e1 \
    --ylim_bottom 1e-14 \
    --plot_sample 100 \
    --plot_size 3 \
    --plot_smooth 0.2 \
    --alpha 0.8 \
    --save \

# McCormick function stochasitc
python main.py --n_iter 40000 \
    --lr_y 0.01 \
    --r 0.01 \
    --func McCormick \
    --grad_noise_y 1e-2 \
    --grad_noise_x 1e-2 \
    --plot_sample 100 \
    --plot_size 2 \
    --plot_smooth 0.8 \
    --ylim_top 1e2 \
    --ylim_bottom 5e-3 \
    --alpha 0.8 \
    --save \

python main.py --n_iter 15000 \
    --lr_y 0.01 \
    --r 0.03 \
    --func McCormick \
    --grad_noise_y 1e-2 \
    --grad_noise_x 1e-2 \
    --plot_sample 100 \
    --plot_size 2 \
    --plot_smooth 0.8 \
    --ylim_top 1e1 \
    --ylim_bottom 3e-3 \
    --alpha 0.8 \
    --save \

python main.py --n_iter 23000 \
    --lr_y 0.01 \
    --r 0.05 \
    --func McCormick \
    --grad_noise_y 1e-2 \
    --grad_noise_x 1e-2 \
    --plot_sample 100 \
    --plot_size 2 \
    --plot_smooth 0.8 \
    --ylim_top 1e1 \
    --ylim_bottom 3e-3 \
    --alpha 0.8 \
    --save \
