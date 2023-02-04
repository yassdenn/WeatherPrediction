// JavaScript

import * as tf from '@tensorflow/tfjs';
import pkg from 'scikitjs'
const { MinMaxScaler } = pkg

console.log('loading');
const model = await tf.loadLayersModel('https://wettermodel.s3.eu-central-1.amazonaws.com/model.json');
model.summary();

const data = [13.6,91.,12.2, 1002.8,8.,12.4,85.,9.9,1003.8,8.,11.4,92.,10.1,1003.,8.,11.2,96.,10.6,1003.8,8.,11.2,96.,10.6,1004.3,8.,10.9,94.,10.,1004.7,7.9100728]

async function scaleData(data) {
  try {
    const scaler = new MinMaxScaler();
    const scaledData = scaler.fitTransform(data);
    return scaledData;
  } catch (err) {
    console.error(`Failed to scale data: ${err}`);
    throw err;
  }
}

scaleData(data).then((scaledData) => {
  // Use the scaled data
  console.log(scaledData)
  const imput_vals = tf.tensor3d(scaledData)
});


const values = tf.tensor3d([[[0.50957854, 0.89534884, 0.72207084, 0.35868006, 0.29166667,0.48659004, 0.8255814 , 0.65940054, 0.37302726, 0.29166667,0.46743295, 0.90697674, 0.66485014, 0.3615495 , 0.29166667,0.46360153, 0.95348837, 0.67847411, 0.37302726, 0.29166667,0.46360153, 0.95348837, 0.67847411, 0.38020086, 0.29166667,0.45785441, 0.93023256, 0.66212534, 0.38593974]]]);

const pred = model.predict(values).dataSync();

console.log(pred)
