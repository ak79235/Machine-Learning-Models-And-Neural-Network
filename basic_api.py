from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
from PIL import Image
import cv2
import keras
from joblib import Parallel, delayed
import joblib
from keras.models import load_model

app = Flask(__name__)
new_model = load_model('C:\\Users\\ak792\\Desktop\\Yethi\\captcha_OCR\\saved_model\\my_model')
new_model.summary()

def decode_batch_predictions(pred):
    pred = pred[:, :-2]
    input_len = np.ones(pred.shape[0])*pred.shape[1]
    
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred,input_length=input_len,greedy=True)[0][0]
    
    # Iterate over the results and get back the text
    output_text = []
    characters=['2', '3', '4', '5', '6', '7', '8', 'b', 'c', 'd', 'e', 'f', 'g', 'm', 'n', 'p', 'w', 'x', 'y']
    labels_to_char={0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: 'b', 8: 'c', 9: 'd', 10: 'e', 11: 'f', 12: 'g', 13: 'm', 14: 'n', 15: 'p', 16: 'w', 17: 'x', 18: 'y'}
    for res in results.numpy():
        outstr = ''
        for c in res:
            if c < len(characters) and c >=0:
                outstr += labels_to_char[c]
        output_text.append(outstr)
    
    # return final text results
    return output_text

def prediction():
    class DataGenerator(keras.utils.Sequence):
        def __init__(self,
                    data,
                    batch_size=16,
                    img_width=200,
                    img_height=50,
                    downsample_factor=4,
                    max_length=5,
                    shuffle=True
                    ):
            self.data = data
            self.batch_size = batch_size
            self.img_width = img_width
            self.img_height = img_height
            self.downsample_factor = downsample_factor
            self.max_length = max_length
            self.shuffle = shuffle
            self.indices = np.arange(len(data))    
            self.on_epoch_end()
            
        def __len__(self):
            return int(np.ceil(len(self.data) / self.batch_size))
        
        def __getitem__(self, idx):
            # 1. Get the next batch indices
            curr_batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
            
            # 2. This isn't necessary but it can help us save some memory
            # as not all batches the last batch may not have elements
            # equal to the batch_size 
            batch_len = len(curr_batch_idx)
            
            # 3. Instantiate batch arrays
            batch_images = np.ones((batch_len, self.img_width, self.img_height, 1),
                                dtype=np.float32)
            input_length = np.ones((batch_len, 1), dtype=np.int64) * \
                                    (self.img_width // self.downsample_factor - 2)
            
            
            for j, idx in enumerate(curr_batch_idx):
                img = self.data[idx].T
                img = np.expand_dims(img, axis=-1)
                batch_images[j] = img
            
            batch_inputs = {
                    'input_data': batch_images,
                    'input_length': input_length,
                    }
            return batch_inputs, np.zeros(batch_len).astype(np.float32)
            
        
        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.indices)
    
    
    batch_size = 16
    # Desired image dimensions
    img_width=200
    img_height=50 

    # Factor  by which the image is going to be downsampled
    # by the convolutional blocks
    downsample_factor=4

    # Maximum length of any captcha in the data
    max_length=5
    
    img=cv2.imread("C:\\Users\\ak792\\Desktop\\Yethi\\3cpwb.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img, (200,50))
    img = (img/255.).astype(np.float32)  
    
    
    train_data_generator = DataGenerator(data=[img],
                                     batch_size=batch_size,
                                     img_width=img_width,
                                     img_height=img_height,
                                     downsample_factor=downsample_factor,
                                     max_length=max_length,
                                     shuffle=True
                                    )
    
    
    print(train_data_generator)
    
    for p, (inp_value, _) in enumerate(train_data_generator):
            X_data = inp_value['input_data']
            z=new_model.predict([X_data[0].reshape(1, 200, 50, 1)])
            return(decode_batch_predictions(z))








@app.route('/apii/', methods=['POST'])
def login():
    # event = json.loads(request.data)
    # return event.get('username')
    try: 
        output=prediction()
        return output
    except IOError:
        return "fail"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')