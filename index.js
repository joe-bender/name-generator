// Constants
const sequenceLength = 15;
const lowercaseLetters = 'abcdefghijklmnopqrstuvwxyz'
const chars = lowercaseLetters + ' ';
let epsilon;
let choices;
let startingLetter = 'b';
let numNames = 20;

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
    let argmaxes = probsToArgmaxes(probs.dataSync());
    // epsilon greedy letter choice
    let letterInt;
    if (Math.random() < epsilon) {
      // choose randomly from the first n argmaxes
      letterInt = _.sample(argmaxes.slice(0, choices));
    } else {
      // choose the first argmax
      letterInt = argmaxes[0];
    }
    let letter = iToChar(letterInt);
    if (letter == ' ') {
      break;
    }
    name += letter;
  }
  // capitalize name
  return name.charAt(0).toUpperCase() + name.slice(1);
}

function probsToArgmaxes(probs) {
  let argmaxes = [];
  let max_prob = -1;
  let max_i = -1;
  for (let i_outer = 0; i_outer < probs.length; i_outer++) {
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] > max_prob) {
        max_prob = probs[i];
        max_i = i;
      }
    }
    argmaxes.push(max_i)
    probs[max_i] = -1;
    max_prob = -1;
    max_i = -1;
  }
  return argmaxes;
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
    // add the generate button
    let button = $('<button type="button" id="genButton">Generate Names</button>');
    let controls = $('#controls');
    controls.prepend(button);

    let loadingDisplay = $('#loadingDisplay');
    loadingDisplay.hide();

    let nameDisplay = $('#nameDisplay');


    let epsilonSlider = $('#epsilonSlider');
    epsilon = epsilonSlider.val();
    let epsilonDisplay = $('#epsilonDisplay');
    epsilonDisplay.text(epsilon);
    epsilonSlider.on('input', function(){
      epsilon = epsilonSlider.val();
      epsilonDisplay.text(epsilon);
    });

    let choicesSlider = $('#choicesSlider');
    choices = choicesSlider.val();
    let choicesDisplay = $('#choicesDisplay');
    choicesDisplay.text(choices);
    choicesSlider.on('input', function(){
      choices = choicesSlider.val();
      choicesDisplay.text(choices);
    });

    function toLoadingMode() {
      button.hide();
      loadingDisplay.show();
      nameDisplay.hide()
      nameDisplay.empty();
    }

    async function appendNames() {
      let listItems = '';
      for (let i = 0; i < numNames; i++) {
        let char = _.sample(lowercaseLetters);
        let name = genName(model, char);
        listItems += '<li>'+name+'</li>';
      }
      nameDisplay.html(listItems);
    }

    function toDisplayMode() {
      button.show();
      loadingDisplay.hide();
      nameDisplay.show();
    }

    button.click(function(){
      toLoadingMode();
      setTimeout(function(){
        appendNames().then(toDisplayMode);
      }, 1);
    });

  });
});
