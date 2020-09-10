let videoWidth, videoHeight,
  fingerLookupIndices = {
    thumb: [0, 1, 2, 3, 4],
    indexFinger: [0, 5, 6, 7, 8],
    middleFinger: [0, 9, 10, 11, 12],
    ringFinger: [0, 13, 14, 15, 16],
    pinky: [0, 17, 18, 19, 20]
  }; // for rendering each finger as a polyline

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 500;

const state = {
  backend: 'webgl',
  showKeypoints: true,
  boundingboxdetection: {
    widthScaleFactor: 2,
    heightScaleFactor: 2,
    scoreThreshold: 0.6,
    maxNumBoxes: 2,
    iouThreshold: 0.5,
    showBoundingBoxes: true,
  },
};

function setupDatGui() {
  const gui = new dat.GUI({ width: 300 });

  let boundingbox = gui.addFolder('Bounding Boxes');
  boundingbox.add(state.boundingboxdetection, 'scoreThreshold')
    .min(0)
    .max(1)
    .step(0.01)
    .onChange(async s => {
      load('Changing score threshold to ' + s.toFixed(2).toString() + '...');
      model2Params.scoreThreshold = s;
      model2 = await handTrack.load(model2Params);
      resume();
    });
  boundingbox.add(state.boundingboxdetection, 'maxNumBoxes')
    .min(1)
    .max(20)
    .step(1)
    .onChange(async b => {
      load('Changing maximum number of boxes to ' + b.toFixed(0).toString() + '...');
      model2Params.maxNumBoxes = b;
      model2 = await handTrack.load(model2Params);
      resume();
    });
  boundingbox.add(state.boundingboxdetection, 'iouThreshold')
    .min(0)
    .max(1)
    .step(0.1)
    .onChange(async i => {
      load('Changing IoU Threshold to ' + i.toFixed(1).toString() + '...');
      model2Params.iouThreshold = i;
      model2 = await handTrack.load(model2Params);
      resume();
    });
  boundingbox.add(state.boundingboxdetection, 'widthScaleFactor')
    .min(1)
    .max(5)
    .step(0.5)
    .onChange(async w => {
      boundingboxParams.width = w;
    });
  boundingbox.add(state.boundingboxdetection, 'heightScaleFactor')
    .min(1)
    .max(5)
    .step(0.5)
    .onChange(async h => {
      boundingboxParams.height = h;
    });
  boundingbox.add(state.boundingboxdetection, 'showBoundingBoxes');

  boundingbox.open();

  let joints = gui.addFolder('Hand Keypoints');
  joints.add(state, 'backend', ['webgl', 'cpu'])
    .onChange(async backend => {
      load('Changing backend to ' + backend + '...');
      await tf.setBackend(backend);
      resume();
    });
  joints.add(state, 'showKeypoints');
  joints.open();
}

function drawPoint(ctx, y, x, r) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fill();
}

function drawKeypoints(ctx, keypoints) {
  if (state.showKeypoints) {
    const keypointsArray = keypoints;

    for (let i = 0; i < keypointsArray.length; i++) {
      const y = keypointsArray[i][0];
      const x = keypointsArray[i][1];
      drawPoint(ctx, x - 2, y - 2, 3);
    }

    const fingers = Object.keys(fingerLookupIndices);
    for (let i = 0; i < fingers.length; i++) {
      const finger = fingers[i];
      const points = fingerLookupIndices[finger].map(idx => keypoints[idx]);
      drawPath(ctx, points, false);
    }
  };
  return;
}

function drawBoundingBox(ctx, r) {
  if (state.boundingboxdetection.showBoundingBoxes) {
    ctx.beginPath();
    ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
    ctx.fillRect(r.bbox[0], r.bbox[1] - 17, r.bbox[2], 17);
    ctx.rect(...r.bbox);
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#0063FF";
    ctx.fillStyle = "#0063FF";
    ctx.fillRect(r.bbox[0] + r.bbox[2] / 2, r.bbox[1] + r.bbox[3] / 2, 5, 5);
    ctx.stroke();
    ctx.fillText(r.score.toFixed(3) + "  | hand", r.bbox[0] + 5, 10 < r.bbox[1] ? r.bbox[1] - 5 : 10);
  };
  return;
}

function scaleBoundingBox(canvas, predictions) {
  for (let r = 0; r < predictions.length; r++) {
    let x = predictions[r].bbox[0];
    let y = predictions[r].bbox[1];
    let width = predictions[r].bbox[2];
    let height = predictions[r].bbox[3];
    let updated_x, updated_y, updated_width, updated_height;

    if ((x + (width / 2)) - ((width / 2) * boundingboxParams.width) > 0) {
      updated_x = (x + (width / 2)) - ((width / 2) * boundingboxParams.width);
    } else {
      updated_x = 0;
    }
    if ((y + (height / 2)) - ((height / 2) * boundingboxParams.height) > 0) {
      updated_y = (y + (height / 2)) - ((height / 2) * boundingboxParams.height);
    } else {
      updated_y = 0;
    }
    if (updated_x + (width * boundingboxParams.width) > canvas.width) {
      updated_width = canvas.width - updated_x;
    } else {
      updated_width = width * boundingboxParams.width;
    }
    if (updated_y + (height * boundingboxParams.height) > canvas.height) {
      updated_height = canvas.height - updated_y;
    } else {
      updated_height = height * boundingboxParams.height;
    }

    predictions[r].bbox[0] = updated_x;
    predictions[r].bbox[1] = updated_y;
    predictions[r].bbox[2] = updated_width;
    predictions[r].bbox[3] = updated_height;
  }
}

function drawPath(ctx, points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

let model;
let model2 = null;

const model2Params = {
  flipHorizontal: true, // flip e.g for video
  imageScaleFactor: 1, // reduce input image size for gains in speed.
  maxNumBoxes: 20, // maximum number of boxes to detect
  iouThreshold: 0.5, // ioU threshold for non-max suppression
  scoreThreshold: 0.6, // confidence threshold for predictions.
}

var boundingboxParams = {
  width: 2,
  height: 2,
}

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: VIDEO_WIDTH,
      height: VIDEO_HEIGHT
    },
  });

  video.srcObject = stream;

  handTrack.startVideo(video, stream);

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();
  return video;
}

const main =
  async() => {
    await tf.setBackend(state.backend);
    model = await handpose.load();
    model2 = await handTrack.load(model2Params);
    let video;

    load('Accessing camera...');
    try {
      video = await loadVideo();
    } catch (e) {
      nocamera();
      throw e;
    }

    landmarksRealTime(video);
  }

const landmarksRealTime = async(video) => {
  load('Setting up...');
  setupDatGui();

  const stats = new Stats();
  stats.showPanel(0);
  document.body.appendChild(stats.dom);

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;

  const canvas = document.getElementById('output');

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  const ctx = canvas.getContext('2d');

  video.width = videoWidth;
  video.height = videoHeight;

  ctx.clearRect(0, 0, videoWidth, videoHeight);
  ctx.strokeStyle = 'red';
  ctx.fillStyle = 'red';

  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  // These anchor points allow the hand pointcloud to resize according to its
  // position in the input.
  const ANCHOR_POINTS = [
    [0, 0, 0],
    [0, -VIDEO_HEIGHT, 0],
    [-VIDEO_WIDTH, 0, 0],
    [-VIDEO_WIDTH, -VIDEO_HEIGHT, 0]
  ];

  ctx.font = "10px Arial";

  async function renderPredictions(predictions_boxes, canvas, ctx, video) {
    ctx.drawImage(video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width, canvas.height);

    scaleBoundingBox(canvas, predictions_boxes);

    var predictions_joints = [];
    let annots = ['indexFinger', 'middleFinger', 'pinky', 'ringFinger', 'thumb'];
    let pointees = ['_tip', '_upperjoint', '_midjoint', '_lowerjoint'];

    const promises_1 = predictions_boxes.map(async i => {
      const outputImage = document.createElement('canvas');
      outputImage.width = i.bbox[2];
      outputImage.height = i.bbox[3];
      const ctx2 = outputImage.getContext('2d');
      ctx2.drawImage(canvas, i.bbox[0], i.bbox[1], i.bbox[2], i.bbox[3], 0, 0, i.bbox[2], i.bbox[3]);
      var prediction_element = await model.estimateHands(outputImage);
      if (prediction_element.length > 0) {
        for (let j = 0; j < annots.length; j++) {
          for (let k = 0; k < pointees.length; k++) {
            prediction_element[0]['annotations'][annots[j]][k][0] += i.bbox[0];
            prediction_element[0]['annotations'][annots[j]][k][1] += i.bbox[1];
          }
        }
        prediction_element[0]['annotations']['palmBase'][0][0] += i.bbox[0];
        prediction_element[0]['annotations']['palmBase'][0][1] += i.bbox[1];
        predictions_joints.push(prediction_element);
      }
      return;
    });

    ctx.save();
    model2Params.flipHorizontal && (ctx.scale(-1, 1), ctx.translate(-video.width, 0));
    const promises_2 = predictions_boxes.map(async r => {
      drawBoundingBox(ctx, r);
      return;
    })

    await Promise.all(promises_1);
    const promises_3 = predictions_joints.map(async i => {
      drawKeypoints(ctx, i[0].landmarks, i[0].annotations);
      return;
    });

    await Promise.all(promises_2);
    await Promise.all(promises_3);
    ctx.restore();
  };

  async function runDetection() {
    stats.begin();
    const predictions_boxes = await model2.detect(video);
    renderPredictions(predictions_boxes, canvas, ctx, video);
    stats.end();
    requestAnimationFrame(runDetection);
  };

  runDetection();
  resume();
};

navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

main();

const canvas_wrapper = document.getElementById('canvas-wrapper');
const loading = document.getElementById('loading');
const setting_gear_image = document.getElementById('setting_gear_image');
const setting_gear_text = document.getElementById('setting_gear_text');

function load(message) {
  setting_gear_text.innerText = message;
  loading.style.visibility = "visible";
  canvas_wrapper.style.visibility = "hidden";
}

function resume() {
  loading.style.visibility = "hidden";
  canvas_wrapper.style.visibility = "visible";
}

function nocamera() {
  setting_gear_image.style.visibility = "hidden";
  setting_gear_text.innerText = 'No camera detected!';
  setting_gear_text.style.color = 'red';
}