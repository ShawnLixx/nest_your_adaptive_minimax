for eps in 0.1 0.05 0.02
do

python mnist.py  \
        --lr_x 0.001 \
        --lr_y 0.001 \
        --epsilon "$eps" \
        --n_epoch 200

python mnist.py  \
        --our \
        --lr_x 0.001 \
        --lr_y 0.001 \
        --epsilon "$eps" \
        --n_epoch 200

python mnist.py  \
        --lr_x 0.001 \
        --lr_y 0.01 \
        --epsilon "$eps" \
        --n_epoch 200

python mnist.py  \
        --our \
        --lr_x 0.001 \
        --lr_y 0.01 \
        --epsilon "$eps" \
        --n_epoch 200

done
