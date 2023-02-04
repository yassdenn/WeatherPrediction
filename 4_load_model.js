// JavaScript

import * as tf from '@tensorflow/tfjs';

console.log('loading');

const model = await tf.loadLayersModel('https://wettermodel.s3.eu-central-1.amazonaws.com/model.json');
model.summary()
