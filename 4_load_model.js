// JavaScript

const tf = require('@tensorflow/tfjs');
const sk = require('scikitjs');
sk.setBackend(tf);

async function load_model() {
  console.log('loading');
  let model = await tf.loadLayersModel('https://wetter1.s3.eu-central-1.amazonaws.com/model.json');
  model.summary();
  return model;
}

const model = load_model();


const data = [4.800000000000001, 72.0, 1029.1, 1.0, 3.8000000000000003, 76.0, 1029.5, 1.0, 3.6999999999999997, 74.0, 1029.5, 1.0, 0.7000000000000002, 86.0, 1029.5, 1.0, -0.7000000000000002, 93.0, 1029.6, 1.0, -0.9000000000000005, 95.0, 1029.5];

const fs = require('fs');

async function loadScaler(filename) {
  try {
    const params = JSON.parse(fs.readFileSync(filename, 'utf-8'));
    console.log(`Scaler loaded from ${filename}`);
    return params;
  } catch (err) {
    console.error(`Failed to load scaler from ${filename}: ${err}`);
    throw err;
  }
}

async function scaleData(params, data) {
  try {
    const min = params.min;
    //console.log(min);
    const range = params.range;
    //console.log(range);
    // Handle the case where data is a one-dimensional array
    const isOneDimensional = !Array.isArray(data[0]);
    const rows = isOneDimensional ? [data] : data;

    // Implement MinMaxScaler in JavaScript
    const scaledData = rows.map((row) => {
      return row.map((value, index) => {
        return (value - min[index]) / range[index];
      });
    });

    return isOneDimensional ? scaledData[0] : scaledData;
  } catch (err) {
    console.error(`Failed to scale data: ${err}`);
    throw err;
  }
}


loadScaler('minmax_scaler1.json').then((params) => {
  scaleData(params, data).then((scaledData) => {
    // Use the scaled data
    console.log(scaledData);
    const val = tf.tensor3d([[scaledData]]);
    model.then(function(pred){
      const prediction = pred.predict(val).dataSync();
      const scaledPred = prediction*24+1
      console.log(Math.round(scaledPred))
      //Use prediction


    });
  });
});

//console.log(scaledData)

//function scaleData(data) {
//  let minValue = Math.min(...data);
//  let maxValue = Math.max(...data);
//  let range = maxValue - minValue;

//  return data.map((x) => (x - minValue) / range);
//};
//const scaledData = scaleData(data);
//console.log(scaledData);

