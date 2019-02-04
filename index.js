// Constants
const sequenceLength = 15;
const lowercaseLetters = 'abcdefghijklmnopqrstuvwxyz'
const chars = lowercaseLetters + ' ';

async function loadModel() {
  let model = await tf.loadModel('models/model.json');
  return model;
}

function genName(model, firstLetter) {
  let name = firstLetter;
  for (let i_seq = 0; i_seq < sequenceLength; i_seq++) {
    let x = nameToX(name);
    let y_pred = model.predict(x);
    let probs = y_pred.squeeze().slice([i_seq, 0], [1, chars.length]).squeeze();
    let letterInt = probs.argMax().dataSync()[0];
    let letter = iToChar(letterInt);
    if (letter == ' ') {
      break;
    }
    name += letter;
    console.log(name);
  }
  // capitalize name
  return name.charAt(0).toUpperCase() + name.slice(1);
}

function probsToSortedArgmaxes(probs) {

}

function charToI(char) {
  return chars.indexOf(char);
}

function iToChar(i) {
  return chars[i];
}

function nameToX(name) {
  let namePadded = name.padEnd(sequenceLength);
  let nameInts = [];
  for (let char of namePadded) {
    nameInts.push(charToI(char));
  }
  let nameTensor = tf.tensor2d([nameInts])
  return nameTensor;
}

$(document).ready(function() {
  loadModel().then(function(model) {
    let nameDisplay = $('p#nameDisplay');
    let button = $('#genButton');
    button.click(function(){
      let char = _.sample(lowercaseLetters);
      let name = genName(model, char);
      nameDisplay.text(name);
    });
  });
});
