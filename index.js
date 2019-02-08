// Constants
const maxLength = 15;
const lowercaseLetters = 'abcdefghijklmnopqrstuvwxyz';
const chars = lowercaseLetters + ' ';
let epsilon;
let choices;
let numNames = 10;

async function loadModel() {
  let model = await tf.loadModel('models/model.json');
  return model;
}

function genName(model, firstLetter) {
  let name = firstLetter;
  for (let i_seq = 0; i_seq < maxLength; i_seq++) {
    let x = nameToX(name);
    let y_pred = model.predict(x);
    let probs = y_pred.slice([0, i_seq]).flatten();
    let argmaxes = probsToArgmaxes(probs.dataSync());
    // epsilon greedy letter choice
    let letterInt;
    if (Math.random() < epsilon) {
      // choose randomly from the first n argmaxes
      letterInt = _.sample(argmaxes);
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
  // for (let i_outer = 0; i_outer < probs.length; i_outer++) {
  for (let i_outer = 0; i_outer < choices; i_outer++) {
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] > max_prob) {
        max_prob = probs[i];
        max_i = i;
      }
    }
    argmaxes.push(max_i);
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
  let nameInts = [];
  for (let char of name) {
    nameInts.push(charToI(char));
  }
  let nameTensor = tf.tensor2d([nameInts]);
  return nameTensor;
}

$(document).ready(function() {

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

  let button = $('#genButton');
  button.hide();
  let loadingDisplay = $('#loadingDisplay');

  let nameDisplay = $('#nameDisplay');

  setTimeout(function() {
    loadModel().then(function(model) {
      button.show();
      loadingDisplay.hide();

      function toLoadingMode() {
        button.hide();
        loadingDisplay.show();
        nameDisplay.hide();
        nameDisplay.empty();
      }

      async function appendNames() {
        let listItems = '';
        for (let i = 0; i < numNames; i++) {
          let char = _.sample(lowercaseLetters);
          let name = genName(model, char);
          listItems += '<li class="list-group-item py-1">' + name + '</li>';
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
        }, 100);
      });
    });
  }, 100);
});
