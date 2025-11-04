function inRange(i, min, max) {
  return i >= min && i <= max;
}
function isDigit(i) {
  return inRange(i, 48, 57);
}
function isLetter(i) {
  return inRange(i, 65, 90) || inRange(i, 97, 122);
}

function exportAlphabet(font, fontname, numPoints = 100) {
  textAlign(CENTER, CENTER);

  const initSF = parseInt(numPoints / 100);
  const ttpOptions = { sampleFactor: initSF };
  const fontSize = 48;

  let csvStr = "";

  for (let i = 48; i < 123; i++) {
    if (isDigit(i) || isLetter(i)) {
      const mchar = String.fromCharCode(i);

      ttpOptions.sampleFactor = initSF;
      let points = font.textToPoints(mchar, 0, 0, fontSize, ttpOptions);

      while (abs(points.length - numPoints) > 2) {
        if (points.length < numPoints) {
          ttpOptions.sampleFactor *= 1.02;
        } else {
          ttpOptions.sampleFactor *= 0.98;
        }

        points = font.textToPoints(mchar, 0, 0, fontSize, ttpOptions);
      }

      csvStr += `${fontname},${mchar}`;

      for (let p of points) {
        csvStr += `,${p.x},${p.y}`;
      }
      csvStr += "\n";
    }
  }

  console.log(`${fontname} done ${int(millis() / 1000)}`);
  return csvStr;
}

const FONTS = {
  baskerville: "./fonts/LibreBaskerville-Regular.ttf",
  cinzel: "./fonts/Cinzel-Regular.ttf",
  garamond: "./fonts/EBGaramond-Regular.ttf",
  habibi: "./fonts/Habibi-Regular.ttf",
  inter: "./fonts/Inter-Regular.ttf",
  lato: "./fonts/Lato-Regular.ttf",
  montserrat: "./fonts/Montserrat-Regular.ttf",
  newsreader: "./fonts/Newsreader-Regular.ttf",
  notosans: "./fonts/NotoSans-Regular.ttf",
  opensans: "./fonts/OpenSans-Regular.ttf",
  playfair: "./fonts/PlayfairDisplay-Regular.ttf",
  ptserif: "./fonts/PTSerif-Regular.ttf",
  roboto: "./fonts/Roboto-Regular.ttf",
  ubuntu: "./fonts/Ubuntu-Regular.ttf",
};

let startButt, saveButt;

function setup() {
  createCanvas(400, 400);
  background(220);
  noLoop();

  startButt = createButton("START");
  startButt.position(10, 10);
  startButt.mouseClicked(startProcessing);

  saveButt = createButton("SAVE");
  saveButt.position(100, 10);
  saveButt.mouseClicked(saveResults);
  saveButt.hide();
}

let csvResult = "font,char";
const NUM_POINTS = 384;

function startProcessing() {
  startButt.hide();

  for (let i = 0; i < NUM_POINTS + 2; i++) {
    csvResult += `,x${i},y${i}`;
  }
  csvResult += "\n";

  for (const fontname in FONTS) {
    loadFont(FONTS[fontname], (font) => {
      csvResult += exportAlphabet(font, fontname, NUM_POINTS);
    });
  }
  saveButt.show();
}

function saveResults() {
  saveStrings([csvResult], `fonts_${NUM_POINTS}_p5_raw.csv`);
}
