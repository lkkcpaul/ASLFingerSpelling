# ASLFingerSpelling competition

This is our work on the Kaggle competition "American Sign Language Fingerspelling Recognition" (https://www.kaggle.com/competitions/asl-fingerspelling).
The model learns to transcribe videos of American sign language (ASL) fingerspelling into texts. Fingerspelling means spelling out phrases such as phone numbers or websites letter by letter. 
The videos are already preprocessed by mediapipe, which turns the video into landmarks, i.e. coordinates of body parts. 
The data consists of 67208 videos with 46478 unique phrases. 
A constraint of the compeition is that the tflite model must be <40mb, and inference time needs to be around less than 0.6s per video. 
The metric for the competition uses levenshtein distance (a.k.a edit distance). Specifically, it is 1 - (total Levenshtein distance/total phrase length).
Our model received a score of 0.677

# Content of repo
- CreateTfrecords.ipynb converts the landmark data from parquet files to tfrecords, with only relevant landmark points retained.
- AslfsModel9*.ipynb builds and trains the model, and outputs the model weights.
- AslfsModel9-tflite.ipynb converts the tensorflow model to tflite model for speed and portability.

# Model Architecture
The model adopts the encoder-decoder architecture.
The sequence of landmarks is transformed into an embedding space by going through a dense layer, three (causal depth-wise CNN + ECA) blocks, and then added with positional encoding.
The embedded input sequence then goes through 6 layers of encoder blocks, which consist of 3 local causal self attentions followed by a global self attention and MLP. 
On the decoder side, the phrase is tokenized at a character level and goes through an embedding layer + positional encoding, follwed by 3 classic decoder blocks, which consist of causal self attention, cross attention to the ecoded landmark sequence, and MLP. 
The final layer is a dense layer with 62 outputs, which correspond to the logits of the character prediction.
In each block, there is always a gradient bypass where the input of the block is added back to the output of the block. 
We chose batch normalization for the landmark embedding and layer normalization for the other blocks, and the normalizations are done before the block transformations, as suggested by this paper https://arxiv.org/pdf/2002.04745.pdf.
The nan frames and the paddings are masked, and the attention mechanism ignores them (on top of causal mask).

# Training
The loss is calculated using sparse categorical cross entropy. The model predicts the next character using the information of the landmarks and the characters before. 
The optimizer used is rectified Adam with look ahead. The learning rate has 5 epochs of warmup and has a cosine decay over time. The weight decays proportionally with the learning rate. 
We trained this on 8-core tpu, and the loss converges around 50 epochs, which takes around 2 hours. 
The metric used for determining the best model is masked accuracy (where the padded tokens don't contribute).

# Data preprocessing and augmentation
We normalized the landmarks and concatenated with it the displacements of the landmarks in one and two frames, respectively.
The training data is augmented by shifting, scaling, rotation, random frame dropping, and flipping left and right. 
The landmarks are padded to maximum lengths, as are the encoded target phrases.

