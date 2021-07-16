Adding your custom model:

first, we need to create a block inside *def load_model(self)* with *elif* statment 

```python
elif self.params['model'] == "CNN":
     embedding_layer = Embedding(self.nb_words,
                                        self.params['embedding_dim'],
                                        weights=[self.embedding_matrix],
                                        input_length=self.params['max_seq_len'],
                                        trainable=self.params['embedding_trainable'],
					mask_zero=False,
					name="features")
            
	               
	     
	    sequence_1_input = Input(shape=(self.params['max_seq_len'],), dtype='int32')
	    embedded_sequences_1 = embedding_layer(sequence_1_input)
	
	    sequence_2_input = Input(shape=(self.params['max_seq_len'],), dtype='int32')
	    embedded_sequences_2 = embedding_layer(sequence_2_input)

	    embedded_sequences = concatenate([embedded_sequences_1,embedded_sequences_2])
		
	    # kimCNN 
	    conv1 = Conv1D(filters=100, kernel_size=3, padding="same")(embedded_sequences)
	    batch_1 = BatchNormalization()(conv1)
	    act1 = Activation('relu')(batch_1)
	    flat1 = Flatten()(act1)
            
	    ....
	    
	    ...
	    
	   
	    preds = Dense(1, activation='sigmoid')(merged)
           

            model = Model(inputs=[sequence_1_input, sequence_2_input, feats_input], outputs=preds)
            print(model.summary())
         

            model.compile(loss='binary_crossentropy',
                           optimizer='nadam',
                           #optimizer='adam',
                          metrics=['acc'])
```
Then we need to add the configration file *LSTM.json*

```
{
  "train_data_file": "data/train.csv",
  "test_data_file": "data/test.csv",
  "embedding_file": "data/glove.840B.300d.txt",
  "embedding_file_type": "glove",
  "embedding_dim": 300,
  "embedding_trainable": false,
  "max_nb_words": 200000,
  "max_seq_len": 30,
  "re_weight": true,
  "model": "CNN",
  "nb_epoches": 10,
}
}
```
preparing the training data 

```
A sentence , B sentence , 0 or 1 
```


Finally .. your model 

```
python FDCLSTM.py config/CNN.json
