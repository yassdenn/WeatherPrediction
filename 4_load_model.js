// JavaScript

const tf = require('@tensorflow/tfjs');
const sk = require('scikitjs');
sk.setBackend(tf);

async function load_model() {
  console.log('loading');
  let model = await tf.loadLayersModel('https://wettermodel.s3.eu-central-1.amazonaws.com/model.json');
  model.summary();
  return model;
}

const model = load_model();


const data = [15.100000000000001, 76.0, 10.899999999999999, 1016.0, 1.0, 17.4, 65.0, 10.800000000000002, 1016.4, 2.0, 20.4, 55.0, 11.100000000000003, 1017.1000000000001, 1.0, 21.899999999999995, 50.00000000000001, 11.000000000000002, 1017.5, 1.0, 23.3, 45.99999999999999, 11.000000000000002, 1018.1, 3.0, 25.400000000000002, 41.0, 11.2, 1018.3]

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


loadScaler('minmax_scaler.json').then((params) => {
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

console.log('outside')
//console.log(scaledData)

//function scaleData(data) {
//  let minValue = Math.min(...data);
//  let maxValue = Math.max(...data);
//  let range = maxValue - minValue;

//  return data.map((x) => (x - minValue) / range);
//};
//const scaledData = scaleData(data);
//console.log(scaledData);

