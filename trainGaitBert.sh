pip install gdown
# Downloading the pkl file of the dataset over here...
gdown --id 1j4phfJPn3gj6QhgrFy6FZIEvny-8htzB
unzip /kaggle/working/CASIA-B-pkl
git clone https://github.com/Ankit3002/OpenGait.git
mv OpenGait/* ./
rm -rf OpenGait

# download the model checkpoint (weights) of the encoder from here...
gdown --id 1IurwhECWDjkcuD1XwJTB2DBIsYeavZtR

# download the whole model weights over here...
gdown --id 10L-MZcV4cR-cxrNGgSyBM9R-r1N-jTnu


