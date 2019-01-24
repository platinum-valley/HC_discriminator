export CUDA_VISIBLE_DEVICES=0
train_csv="dataset/dataset.csv"
valid_csv="dataset/valid.csv"

epochs=100
batch_size=64
learning_rate=0.01
weight_decay=0.2
dropout_rate=0.0

input_frame=862
input_dim=20
num_encoder_layers=1
encoder_hidden_dim=64
num_decoder_layers=1
decoder_hidden_dim=64
output_dim=64
label_dim=1


python train.py\
    --gpu\
    --batch_size ${batch_size}\
    --epochs ${epochs}\
    --lr ${learning_rate}\
    --weight_decay ${weight_decay}\
    --dropout_rate ${dropout_rate}\
    --train_data ${train_csv}\
    --valid_data ${valid_csv}\
    --input_frame ${input_frame}\
    --input_dim ${input_dim}\
    --num_encoder_layers ${num_encoder_layers}\
    --encoder_hidden_dim ${encoder_hidden_dim}\
    --num_decoder_layers ${num_decoder_layers}\
    --decoder_hidden_dim ${decoder_hidden_dim}\
    --output_dim ${output_dim}\
    --label_dim ${label_dim}
