// JavaScript

const tf = require('@tensorflow/tfjs');
const axios = require('axios');
const fs = require('fs');
//const sk = require('scikitjs');

insert_link = "https://api23wetterstation.pythonanywhere.com/predictions/insert/KPQHYyj4L6qKbULV"

async function load_model() {
  console.log('loading');
  let model = await tf.loadLayersModel('https://wetter1.s3.eu-central-1.amazonaws.com/model.json');
  model.summary();
  return model;
}

const model = load_model();

const fetchListFromAPI = async () => {
  const response = await axios.get('https://api23wetterstation.pythonanywhere.com/predictions/latest');
  const list = response.data;
  return list;
};

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


const log_pred = async () => {
  const myList = await fetchListFromAPI();
  console.log(myList)
  loadScaler('minmax_scaler1.json').then((params) => {
    scaleData(params, myList).then((scaledData) => {
      // Use the scaled data
      console.log(scaledData);
      const val = tf.tensor3d([[scaledData]]);
      model.then(function(pred){
        const prediction = pred.predict(val).dataSync();
        const scaledPred = Math.round(prediction*24+1)
        console.log(scaledPred)
        //Use prediction
        data = {
          "temperature": myList[myList.length -3],
          "humidity": myList[myList.length - 2],
          "pressure": myList[myList.length-1],
          "class": scaledPred
        };
        console.log(data)
        axios.post(insert_link, data )
        .then(response => {
        // handle success
        console.log(response.data);
        })
        .catch(error => {
        // handle error
        console.log(error);
        });
      });
    });
  });
}

log_pred()



//const data = [4.299999999999999, 72.0, 1033.9, 4.0, 4.299999999999999, 73.0, 1033.6, 4.0, 4.200000000000001, 73.0, 1033.7, 4.0, 4.200000000000001, 74.0, 1033.8, 8.0, 3.999999999999999, 76.0, 1034.0, 8.0, 3.8999999999999986, 78.0, 1034.2];

//console.log(scaledData)

//function scaleData(data) {
//  let minValue = Math.min(...data);
//  let maxValue = Math.max(...data);
//  let range = maxValue - minValue;

//  return data.map((x) => (x - minValue) / range);
//};
//const scaledData = scaleData(data);
//console.log(scaledData);

