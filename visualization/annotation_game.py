"""
Class that contains the bulk of the code for creating the annotation game files for the user study.
"""


import os
import zipfile
import shutil
import formulas.parser as Parser
import formulas.utils as FU

import loaders.metadata_loader as MetaData

# def zipdir(path, ziph):
#     # ziph is zipfile handle
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             ziph.write(os.path.join(root, file),
#                        os.path.relpath(os.path.join(root, file),
#                                        os.path.join(path, '..')))


def zip(src, dst, folder):
    zf = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
    abs_src = os.path.abspath(src)
    for dirname, subdirs, files in os.walk(src):
        for filename in files:
            absname = os.path.abspath(os.path.join(dirname, filename))
            arcname = absname[len(abs_src) + 1:]
            zf.write(absname, f"{folder}/{arcname}")
    zf.close()


def clean(filename):
    n_dashes = 0

    for i, character in enumerate(filename):
        if character == '/':
            n_dashes += 1
            if n_dashes == 4:
                return f"images" + filename[i:]

def create_html(neuron_i, map_im_2_activations, threshold, results, mode, indices, access):

    if not os.path.exists(f"results/annotation_game/{neuron_i}_filenames"):
        os.mkdir(f"results/annotation_game/{neuron_i}_filenames/")
    if not os.path.exists(f"results/annotation_game/{neuron_i}_masks"):
        os.mkdir(f"results/annotation_game/{neuron_i}_masks/")

    mode_s = 'Training' if mode == 'train' else 'Test'

    # ------------------------------------------------------------------------------
    # - Before turning to the html code, first collect some necessary information. -
    # ------------------------------------------------------------------------------
    label_mask = FU.compute_composite_mask(Parser.parse(results[neuron_i]['formula']))

    if mode == 'train':
        for i in indices:
            bitmask = map_im_2_activations[i] > threshold
            with open(f'results/annotation_game/{neuron_i}_masks/{i}.txt', 'w') as f:
                line1 = "".join(['1' if x else '0' for x in bitmask.ravel()])
                line2 = "".join(['1' if x else '0' for x in label_mask[i].ravel()]) if access else ""
                f.write(line1 + line2)
        for i in indices:
            with open(f'results/annotation_game/{neuron_i}_filenames/{i}.txt', 'w') as f:
                f.write(clean(MetaData.filename(i)))

        indices = indices[60:]
    else:
        indices = indices[:60]

    html = f"""<html>
  <head>
    <meta content="text/html;charset=utf-8" http-equiv="Content-Type">
    <meta content="utf-8" http-equiv="encoding">
    <script src='{neuron_i}_script.js'></script>
  </head>

  <body class="unitviz">
    <table border="0" cellpadding="0" cellspacing="20">
      <tr>
        <td>
          <p id="c1" style="font-size: 30pt"><b>Filter {neuron_i} {mode_s} Mode</b></p>
          <p id="c2">
            <b>Click</b> on the image to draw your mask <br />
            <b>Q</b> to score the mask and show the solution  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<br />
            {'<b>W</b> (after Q) to hide/show solution <br />' if mode == 'train' else ''}
            <b>Space</b> (after Q) to switch to the next image <br />
            {'<b>A</b> to adopt label mask <br />' if access else ''}
            <b>S</b> to reset drawn mask <br />
            {'<b>D</b> to hide/show label mask <br />' if access else ''}"""

    if mode == 'train':
        html += """
            <b>R</b> to reset <br />
            <b>Arrow Keys</b> to nagivate through images <br />"""
        # html +="""
        #     <b>P</b> to pause/unpause timer"""
    html += """
          </p>
        </td>
        <td>
          <p>  &nbsp; &nbsp; &nbsp;</p>
          <p>  &nbsp; &nbsp; &nbsp;</p>
          <p><span id ="tf1" style="color: black; z-index: 0;"> </span> </p>
          <p><span id ="tf2" style="color: black; z-index: 0;"> </span> </p>
          <p><span id ="tf3" style="color: black; z-index: 0;"> </span> </p>
          <p><span id ="tf4" style="color: black; z-index: 0;"> </span> </p>
        </td>
      </tr>
      <tr>
        <td>
          <div style="position: relative;">
            <span><canvas id="cn" width="400" height="400" style="position: absolute; left: 0; top: 0; z-index: 1;">  </canvas></span>
          </div>
        </td>
      </tr>
    </table>
  </body>
</html>"""

    javascript = f"""var indices = {indices.tolist()};
var neuron_i = {neuron_i};
var canvas;
var context;
var W;
var H;
var img;
{'var labelMask;' if access else ''}
var mask;
var neuronMask;
var filename;
var imageId;
var guesses = 0;
var score = 0;
var latestScore = 0;
var R = [0, 0, 0, 0, 0];
var totalNeuronArea = 0;
var totalLabelArea = 1;
var totalIntersection = 2;
var totalRandomIntersection = 3;
var totalMaxRandomIntersection = 4;
var union = 0;
var tf1;
var ff2;
var scored = false;
var lockQ = true;
var h = '';"""
    if mode == 'test':
        javascript += """
var guesses = 0;
var gameEnded = false;
"""

    javascript += """
window.onload=init;
window.addEventListener('click', draw, false);

async function eL(e) {"""
    if mode == 'test':
        javascript += """
    if (gameEnded) {
       return;
    }"""
    javascript += """
  if (!lockQ && e.keyCode == "32") {
    guesses++;
    setCookie("guesses=" + guesses + ';');
    window.location.reload(false);
  } else if(e.keyCode == "81") {
    lockQ = false;

    await readNeuronMask(imageId);
    if (!scored) {
      scored = true;

      var neuronArea = sum(neuronMask);
      var labelArea = sum(mask);
      var intersection = sum(compute_intersection(mask, neuronMask));
      var factor = 0.75 / 49;

      R[totalNeuronArea] += neuronArea;
      R[totalLabelArea] += labelArea;
      R[totalIntersection] += intersection;
      R[totalRandomIntersection] += (labelArea * neuronArea);
      R[totalMaxRandomIntersection] += (neuronArea * neuronArea);

      var totalUnion = R[totalNeuronArea] + R[totalLabelArea] - R[totalIntersection];
      var imrou = (R[totalIntersection] - factor * R[totalRandomIntersection]) / totalUnion;
      var maximrou = (R[totalNeuronArea] - factor * R[totalMaxRandomIntersection]) / R[totalNeuronArea];
      score = computeImRoU();
      lT = (intersection - (0.75/49) * (neuronArea*labelArea)) / (neuronArea + labelArea - intersection);
      lB = 1 - (0.75/49) * neuronArea;
      latestScore = lT / lB;
      h = getCookie('h');
      if (h == null || h == 'null') {
          h = "";
      }
      h += "" + labelArea + " " + neuronArea + " " + intersection + "/";
      const xhttp = new XMLHttpRequest();"""+f"""
      xhttp.open("POST", "create.php?n=" + neuron_i + "_{mode}_" + guesses + "&history=" + h);""" + """
      xhttp.send();
      tf1.textContent = 'Total Score: ' + Math.round(1000 * score)/1000;
      tf2.textContent = 'Latest Score: ' + Math.round(1000 * latestScore)/1000;
      tf3.textContent = 'Image ' + (guesses+1) + '/' + indices.length;

      for (i = 0; i < 5; i++) {
        setCookie("r" + i + "=" + R[i] + ';');
      }
      setCookie("latestScore=" + latestScore + ";");
      setCookie('h=' + h + ';');\n
    }

    redrawImage();
  }"""
    if access:
        javascript += """ else if (e.keyCode == "65") {
    for (var y = 0; y < 7; y++) {
      for (var x = 0; x < 7; x++) {
        mask[y][x] = labelMask[y][x];
      }
    }
    redrawImage();
  } else if (e.keyCode == "68") {
    var any = false;
    for (y = 0; y < 7; y++) {
      for (x = 0; x < 7; x++) {
        if (labelMask[y][x]) {
          any = true;
        }
      }
    }

    if (any) {
      labelMask = [new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false)];
    } else {
      await readLabelMask(imageId);
    }
    redrawImage();
  } """
    javascript += """ else if(e.keyCode == "83") {
    mask = [new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false)];
    redrawImage();
  }"""

    if mode == 'train':
        javascript += """
 else if(e.keyCode == "82") {
    score = 0;
    setCookie("score=0.0;");
    for (var i = 0; i < 5; i++) {
      R[i] = 0;
      setCookie("r" + i + "=0;");
    }
    tf1.textContent = "Total Score: " + 0;
  } else if (e.keyCode == "69") {
    neuronMask = [new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false)];
    redrawImage();
  } else if(e.keyCode == "37") {
    navigate(-1);
  } else if(e.keyCode == "38") {
    navigate(50);
  } else if(e.keyCode == "39") {
    navigate(1);
  } else if(e.keyCode == "40") {
    navigate(-50);
  } else if (e.keyCode == "87") {
    if (!lockQ) {
      any = false;
      for (y = 0; y < 7; y++) {
        for (x = 0; x < 7; x++) {
          if (neuronMask[y][x]) {
            any = true;
          }
        }
      }

      if (any) {
        neuronMask = [new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false)];
      } else {
        await readNeuronMask(imageId);
      }
      redrawImage();
    }
  }"""

    javascript += """
}

document.addEventListener('keydown', eL);\n"""

    if mode == 'train':
        javascript += """
function navigate(amount) {
  guesses += amount;
  if (guesses < 0) {
    guesses += indices.length;
  } else if (guesses >= indices.length) {
    guesses -= indices.length;
  }
  setCookie("guesses=" + guesses + ';');
  window.location.reload(false);
}        
"""
    if mode == 'test':
        javascript += """

function download(data, filename, type) {
    var file = new Blob([data], {type: type});
    if (window.navigator.msSaveOrOpenBlob) {
        window.navigator.msSaveOrOpenBlob(file, filename);
    } else { // Others
        var a = document.createElement("a"),
        url = URL.createObjectURL(file);
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        setTimeout(function() {
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }, 0);
    }
}
"""

    javascript += """

function computeImRoU() {
  var factor = 0.75 / 49;

  var totalUnion = R[totalNeuronArea] + R[totalLabelArea] - R[totalIntersection];
  var imrou = (R[totalIntersection] - factor * R[totalRandomIntersection]) / totalUnion;
  var maximrou = (R[totalNeuronArea] - factor * R[totalMaxRandomIntersection]) / R[totalNeuronArea];

  return imrou / maximrou;
}

function init() {
    readGuesses();
"""
    if mode == 'test':

        javascript += """  if (guesses == indices.length) {
    gameEnded = true
    document.getElementById('tf2').textContent = 'Thanks for completing the task! Your unique code is:';
    document.getElementById('tf3').textContent = 'INSERT CODE HERE';
    document.getElementById('tf2').style.fontSize = '30pt';
    document.getElementById('tf3').style.fontSize = '30pt';
    document.getElementById('c1').textContent = '';
    document.getElementById('c2').textContent = '';
 } else {     
"""

    javascript +="""
    canvas = document.getElementById('cn');
    context = canvas.getContext('2d');
    W = canvas.width/7;
    H = canvas.height/7;
    tf1 = document.getElementById('tf1');
    tf2 = document.getElementById('tf2');
    
    readRecords();
    tf1.textContent = 'Total Score: ' + Math.round(1000 * computeImRoU()) / 1000;
    if (guesses === 0) {
      tf2.textContent = 'Latest Score: --';
    } else {
      tf2.textContent = 'Latest Score: ' + Math.round(1000 * getCookie('latestScore')) / 1000;
    }
    tf3.textContent = 'Image ' + (guesses+1) + '/' + indices.length;
    loadNext();"""
    if mode == 'test':
        javascript += "\n}"
    javascript += """
}

function setTime() {"""
    if mode == 'test':
        javascript += """
        if (gameEnded) {
            document.getElementById("tf4").innerHTML = "";
            return;
        }
"""
    javascript += """    
}

function getCookie(cname) {""" + f"""
    var name = "{mode}" + "_" + neuron_i + "_" + cname + "=";""" + """
    var decodedCookie = decodeURIComponent(document.cookie);
    var ca = decodedCookie.split(';');

    for(var i = 0; i < ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0) == ' ') {
            c = c.substring(1);
        }
        if (c.indexOf(name) == 0) {
            return c.substring(name.length, c.length);
        }
    }
  return null;
}

function setCookie(value) {
  var d = new Date();
  d.setTime(d.getTime() + (365*24*60*60*1000));
  var expires = "expires="+ d.toUTCString();""" + f"""
  cookieText = "{mode}" + "_" + neuron_i + "_" + value + expires + ";path=/;SameSite=Lax;";""" + """
  document.cookie = cookieText;
}

async function loadNext() {
   mask = [new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false)];
   neuronMask = [new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false)];
   labelMask = [new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false), new Array(7).fill(false)];
   imageId = indices[guesses];
   await readFilename(imageId);""" + f"""
   {'await readLabelMask(imageId);' if access else ''}""" + """
   img = new Image();
   img.src = filename;
   img.onload = redrawImage;
}

function redrawImage() {
     context.clearRect(0, 0, canvas.width, canvas.height);
     context.drawImage(img, 0, 0, canvas.width, canvas.height);
     drawGrid();
}

function draw(e) {"""
    if mode == 'test':
        javascript += """  if (gameEnded) {
    return;
    }
    """
    javascript += """  var pos = getMousePos(canvas, e);

  x = Math.round((pos.x-1)*7/400 - 0.5);
  y = Math.round((pos.y-1)*7/400 - 0.5);

  if(0 <= y && y < 7 && 0 <= x && x < 7) {
    mask[y][x] = !mask[y][x]

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(img, 0, 0, canvas.width, canvas.height);
    
    drawGrid();
  }
}



async function readNeuronMask(i) {
  name = neuron_i + 'm/' + i + '.txt';
  await fetch(name)
  .then(response => response.text())
  .then(text => textToNeuronMask(text))
}

function textToNeuronMask(text) {
  for(y=0; y<7; y++) {
    for(x=0; x<7; x++) {
       neuronMask[y][x] = (text[y*7+x] === '1');
    }
  }
}"""

    if access:
        javascript += """

async function readLabelMask(i) {
    name = neuron_i + 'm/' + i + '.txt';
    await fetch(name)
    .then(response => response.text())
    .then(text => textToLabelMask(text))
}

function textToLabelMask(text) {
  for(y=0; y<7; y++) {
    for(x=0; x<7; x++) {
       labelMask[y][x] = (text[49 + y*7+x] === '1');
    }
  }
}"""

    javascript += """

async function readFilename(i) {
    name = neuron_i + 'f/' + i + '.txt'
    await fetch(name)
    .then(response => response.text())
    .then(text => textToFilename(text))
}

function textToFilename(text) {
   filename = text;
}

function readRecords() {
  for (var i = 0; i < 5; i++) {
    v = getCookie("r" + i);
    if (v == null) {
      R[i] = 0;
    } else {
      R[i] = parseInt(v);
    }
  }
}

function readGuesses() {
  v = getCookie("guesses");
  if (v == null) {
    guesses = 0;
  } else {
    guesses = parseInt(v);
  }
}

function textToAttempts(text) {
  score = parseInt(text);
}


function getMousePos(canvas, evt) { // function that ensures draw() knows what position was clicked on
    var rect = canvas.getBoundingClientRect();
    return {
      x: evt.clientX - rect.left,
      y: evt.clientY - rect.top
    };
}

function drawGrid() { // paint the rectangular grid over the image
  for (y = 0; y <= 7*H; y += H) {
      for (x = 0; x <= 7*W; x += W) {
          context.moveTo(x, 0);
          context.lineTo(x, 7*H);
          context.stroke();
          context.moveTo(0, y);
          context.lineTo(7*W, y);
          context.stroke();
      }
   }
   context.globalAlpha = 0.4
   for(y = 0; y < 7; y++) {
     for(x = 0; x < 7; x++) {
       if(mask[y][x] && neuronMask[y][x]) {
         context.fillStyle = "#FF00FF";
         context.fillRect(x*W, y*H, W, H);
       } else if(mask[y][x] && !neuronMask[y][x]) {
         context.fillStyle = "#0000FF";
         context.fillRect(x*W, y*H, W, H);
       } else if(!mask[y][x] && neuronMask[y][x]) {
         context.fillStyle = "#FF0000";
         context.fillRect(x*W, y*H, W, H);
       }"""

    if access:
        javascript += """ else if (labelMask[y][x]) {
         context.fillStyle = "#AAAAFF";
         context.fillRect(x*W, y*H, W, H);
       } """

    javascript += """
     }
   }
   context.globalAlpha = 1
}

function sum(twoDimArray) {
   return twoDimArray.reduce(function(a,b) { return a.concat(b); })
                     .reduce(function(a,b) { return a + b; })
}

function compute_intersection(a1, a2) {
  return a1.map((b, i) => b.map((x, j) => x && a2[i][j]));
}"""
    with open(f"results/annotation_game/{neuron_i}.html", 'w') as f:
        f.write(html)

    with open(f"results/annotation_game/{neuron_i}_script.js", 'w') as f:
        f.write(javascript)


    zip(f"results/annotation_game/{neuron_i}_filenames/", f"results/annotation_game/{neuron_i}f.zip", f"{neuron_i}f")
    zip(f"results/annotation_game/{neuron_i}_masks/", f"results/annotation_game/{neuron_i}m.zip", f"{neuron_i}m")


    # Deleting an non-empty folder
    shutil.rmtree(f"results/annotation_game/{neuron_i}_filenames/", ignore_errors=True)
    shutil.rmtree(f"results/annotation_game/{neuron_i}_masks/", ignore_errors=True)