function inRange(i, min, max) {
  return i >= min && i <= max;
}
function isDigit(i) {
  return inRange(i, 48, 57);
}
function isLetter(i) {
  return inRange(i, 65, 90) || inRange(i, 97, 122);
}
function dig(v, digits) {
  return parseFloat(v.toFixed(digits));
}

function exportAlphabet(p5font, font, sampleFactor = 32.0, fontSize = 48) {
  textAlign(CENTER, CENTER);
  const alphabet = [];

  for (let i = 48; i < 123; i++) {
    if (isDigit(i) || isLetter(i)) {
      const char = String.fromCharCode(i);
      const points = p5font.textToPoints(char, 0, 0, fontSize, { sampleFactor }).map(({x,y}) => [dig(x, 6), dig(y, 6)]);

      alphabet.push({
        font,
        char,
        points,
      });
    }
  }
  console.log(`${font} done ${int(millis() / 1000)}`);
  return alphabet;
}

const FONTS = {
  aubrey: "Aubrey-Regular.ttf",
  baskerville: "LibreBaskerville-Regular.ttf",
  bodoni: "BodoniModa_48pt-Regular.ttf",
  cinzel: "Cinzel-Regular.ttf",
  dmserif: "DMSerifText-Regular.ttf",
  exo: "Exo-Regular.ttf",
  faculty: "FacultyGlyphic-Regular.ttf",
  fira: "FiraSans-Regular.ttf",
  funnel: "FunnelSans-Regular.ttf",
  garamond: "EBGaramond-Regular.ttf",
  germania: "GermaniaOne-Regular.ttf",
  glassantiqua: "GlassAntiqua-Regular.ttf",
  googlecode: "GoogleSansCode-Regular.ttf",
  grenze: "Grenze-Regular.ttf",
  habibi: "Habibi-Regular.ttf",
  ibmmono: "IBMPlexMono-Regular.ttf",
  ibmsans: "IBMPlexSans-Regular.ttf",
  ibmserif: "IBMPlexSerif-Regular.ttf",
  inconsolata: "Inconsolata-Regular.ttf",
  inter: "Inter-Regular.ttf",
  lato: "Lato-Regular.ttf",
  merriweather: "Merriweather_48pt-Regular.ttf",
  montserrat: "Montserrat-Regular.ttf",
  newsreader: "Newsreader-Regular.ttf",
  notosans: "NotoSans-Regular.ttf",
  opensans: "OpenSans-Regular.ttf",
  oxanium: "Oxanium-Regular.ttf",
  playfair: "PlayfairDisplay-Regular.ttf",
  poppins: "Poppins-Regular.ttf",
  ptserif: "PTSerif-Regular.ttf",
  roboto: "Roboto-Regular.ttf",
  rubik: "Rubik-Regular.ttf",
  sansation: "Sansation-Regular.ttf",
  slabo: "Slabo27px-Regular.ttf",
  tasaexplorer: "TASAExplorer-Regular.ttf",
  tiktoksans: "TikTokSans-Regular.ttf",
  tinos: "Tinos-Regular.ttf",
  ubuntu: "Ubuntu-Regular.ttf",
  vendsans: "VendSans-Regular.ttf",
  zain: "Zain-Regular.ttf",
  zalando: "ZalandoSans-Regular.ttf",
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

const allLetters = [];

async function startProcessing() {
  startButt.hide();

  for (const fontname in FONTS) {
    const font = await loadFont(`./fonts/${FONTS[fontname]}`);
    allLetters.push(...exportAlphabet(font, fontname, 40.0));
  }

  const lens = allLetters.map(l => l["points"].length);
  console.log(allLetters.length);
  console.log(Math.min(...lens), Math.max(...lens));

  saveButt.show();
}

function saveResults() {
  // saveJSON(allLetters, "fonts_p5_raw.json");
  saveStrings([JSON.stringify(allLetters)], "fonts_p5_raw.json");
}
