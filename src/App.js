import * as tf from '@tensorflow/tfjs';
import React, { useEffect, useRef, useState } from 'react';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import * as cocossd from '@tensorflow-models/coco-ssd';
import { initNotifications, notify } from '@mycv/f8-notification';
import { saveAs } from 'file-saver';
import { Howl } from 'howler';
import './App.css';
import alertphone from './assets/alertphone.mp3';
import khongduocdichuyen from './assets/khongduocdichuyen.mp3';
import khongduocquaycop from './assets/khongduocquaycop.mp3';
import hetthoigianlambai from './assets/hetthoigianlambai.mp3';
import batdauthoigianlambai from './assets/batdauthoigianlambai.mp3';

tf.setBackend('webgl');

// Define alert sounds
const alertphonesound = new Howl({
  src: [alertphone],
});

const khongduocdichuyensound = new Howl({
  src: [khongduocdichuyen],
});

const khongduocquaycopsound = new Howl({
  src: [khongduocquaycop],
});

const hetthoigianlambaisound = new Howl({
  src: [hetthoigianlambai],
});

const batdauthoigianlambaisound = new Howl({
  src: [batdauthoigianlambai],
});

const NORMAL_POSTURE_LABEL = 'normal_posture';
const HEAD_LEFT_LABEL = 'head_left';
const HEAD_RIGHT_LABEL = 'head_right';
const STANDING_UP_LABEL = 'standing_up';
const ABSENT_LABEL = 'absent';

const TRAINING_TIMES = 50;
const DETECTION_CONFIDENCE = 0.8;

function App() {
  const video = useRef();
  const canvasRef = useRef(null); // For object detection
  const classifier = useRef();
  const mobilenetModule = useRef();
  const cocoModel = useRef(); // COCO-SSD model reference
  const [currentBehavior, setCurrentBehavior] = useState('');
  const [detections, setDetections] = useState([]);
  const [testTime, setTestTime] = useState(1); // Default test time in minutes
  const [timeRemaining, setTimeRemaining] = useState(0);
  const [isTesting, setIsTesting] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainedModel, setTrainedModel] = useState(null);

  const init = async () => {
    console.log('Initializing...');
    await setupCamera();

    console.log('Camera setup success.');

    classifier.current = knnClassifier.create();
    mobilenetModule.current = await mobilenet.load();
    console.log('MobileNet loaded.');
    cocoModel.current = await cocossd.load(); // Load COCO-SSD model

    console.log('Setup complete.');
    console.log('Train each behavior using the buttons below.');

    initNotifications({ cooldown: 3000 });

    // Start COCO-SSD object detection
    runObjectDetection();
    loadModel();
  };

  const setupCamera = () => {
    return new Promise((resolve, reject) => {
      navigator.getUserMedia = navigator.getUserMedia ||
        navigator.webkitGetUserMedia ||
        navigator.mozGetUserMedia ||
        navigator.msGetUserMedia;

      if (navigator.getUserMedia) {
        navigator.getUserMedia(
          { video: true },
          stream => {
            video.current.srcObject = stream;
            video.current.addEventListener('loadeddata', resolve);
          },
          error => reject(error)
        );
      } else {
        reject();
      }
    });
  };

  const captureFrame = async (label, index) => {
    const canvas = document.createElement('canvas');
    const videoElement = video.current;
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    // Convert canvas to Blob and save as image
    canvas.toBlob((blob) => {
      saveAs(blob, `${index}.png`);
    });
  };

  const train = async (label) => {
    setIsTraining(true);
    console.log(`[ ${label} ] training started...`);
    for (let i = 0; i < TRAINING_TIMES; i++) {
      console.log(`Training progress: ${((i + 1) / TRAINING_TIMES) * 100}%`);
      await captureFrame(label, i);

      const embedding = mobilenetModule.current.infer(video.current, true);
      classifier.current.addExample(embedding, label);

      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    console.log(`[ ${label} ] training complete.`);
    setIsTraining(false);
  };

  const saveModel = async () => {
    const dataset = classifier.current.getClassifierDataset();
    const datasetObj = Object.fromEntries(
      Object.entries(dataset).map(([label, data]) => [label, data.arraySync()])
    );
    const blob = new Blob([JSON.stringify(datasetObj)], { type: 'application/json' });
    saveAs(blob, 'model.json');
    alert('Model saved as model.json');
  };

  const loadModel = async () => {
    const response = await fetch('./model.json');
    const json = await response.json();
    const dataset = Object.fromEntries(
      Object.entries(json).map(([label, data]) => [label, tf.tensor(data)])
    );
    classifier.current.setClassifierDataset(dataset);
    setTrainedModel(classifier.current);
    console.log('Model loaded successfully.');
  };

  const run = async () => {
    console.log(isTesting);
    // if (!isTesting) {
    //   console.log('Testing stopped.');
    //   return; // Stop the loop if testing is no longer active
    // }

    // if (!trainedModel) {
    //   alert('Please train and load the model first!');
    //   return;
    // }

    const embedding = mobilenetModule.current.infer(video.current, true);
    const result = await trainedModel.predictClass(embedding);

    if (result.confidences[result.label] > DETECTION_CONFIDENCE) {
      setCurrentBehavior(result.label);
  
      switch (result.label) {
        case HEAD_LEFT_LABEL:
        case HEAD_RIGHT_LABEL:
          if (!khongduocquaycopsound.playing()) {
            khongduocquaycopsound.play();
          }
          console.log('Head turned detected');
          break;
  
        case STANDING_UP_LABEL:
        case ABSENT_LABEL:
          if (!khongduocdichuyensound.playing()) {
            khongduocdichuyensound.play();
          }
          console.log('Standing up or absent detected');
          break;
  
        default:
          console.log(`Detected: ${result.label}`);
          break;
      }
    }

    requestAnimationFrame(run);
  };

  const runObjectDetection = () => {
    setInterval(async () => {
      if (
        typeof video.current !== "undefined" &&
        video.current !== null &&
        video.current.readyState === 4
      ) {
        // Get video properties
        const videoElement = video.current;
        const videoWidth = videoElement.videoWidth;
        const videoHeight = videoElement.videoHeight;

        // Set canvas dimensions
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;

        // Detect objects
        const objects = await cocoModel.current.detect(videoElement);

        // Update state and draw on canvas
        setDetections(objects);
        const ctx = canvasRef.current.getContext("2d");
        drawRect(objects, ctx);

        // Check for 'cell phone' detection and play alert sound
        objects.forEach(obj => {
          if (obj.class === 'cell phone') {
            if (!alertphonesound.playing()) {
              alertphonesound.play();
            }
          }
        });
      }
    }, 500); // Run every 500ms
  };

  const drawRect = (detections, ctx) => {
    // Clear previous drawings
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    detections.forEach(prediction => {
      const [x, y, width, height] = prediction['bbox'];
      const text = prediction['class'];

      // Random color for each detection
      const color = Math.floor(Math.random() * 16777215).toString(16);
      ctx.strokeStyle = `#${color}`;
      ctx.font = '18px Arial';
      ctx.beginPath();
      ctx.fillStyle = `#${color}`;
      ctx.fillText(text, x, y - 10); // Text above rectangle
      ctx.rect(x, y, width, height);
      ctx.stroke();
    });
  };

  const startTest = () => {
    // Load the model
    loadModel();
    setTimeRemaining(testTime * 60); // Convert minutes to seconds
    setIsTesting(true);
    if (!batdauthoigianlambaisound.playing()) {
      batdauthoigianlambaisound.play();
    }

    try {
      // Start behavior detection
      run();
    } catch (error) {
      console.error("Error during Start Test:", error);
      alert("Failed to start the test. Please ensure the model is loaded and ready.");
      setIsTesting(false);
      return;
    }

    const timer = setInterval(() => {
      setTimeRemaining(prev => {
        if (prev <= 1) {
          clearInterval(timer);
          setIsTesting(false);
          if (!hetthoigianlambaisound.playing()) {
            hetthoigianlambaisound.play();
          }
          // alert('Test is over!');
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    run(); // Start the behavior detection loop
  };

  const sleep = (ms = 0) => {
    return new Promise(resolve => setTimeout(resolve, ms));
  };

  useEffect(() => {
    init();

    // Cleanup
    return () => { };
  }, []);

  return (
    <div className="main">
      {/* Video and Canvas Section */}
      <div className="video-container">
        <video
          ref={video}
          className="video"
          autoPlay
        />
        <canvas
          ref={canvasRef}
          className="canvas"
        />
      </div>

      {/* Controls Section */}
      <div className="control">
        {/* Timer and Duration Settings */}
        <div className="input-group">
          <label htmlFor="test-duration">Test Duration (minutes):</label>
          <input
            id="test-duration"
            type="number"
            min="1"
            value={testTime}
            onChange={e => setTestTime(Number(e.target.value))}
            
            style={{ width: '60px', textAlign: 'center', fontSize: '1.2rem' }}
          />
        </div>
        <button className="btn" onClick={startTest} >
          Start Test
        </button>

        {/* Timer Display */}
        <div className="timer">
          Time Remaining: {`${Math.floor(timeRemaining / 60)}:${String(timeRemaining % 60).padStart(2, '0')}`}
        </div>

        {/* Training Buttons */}
        {/*
        <div className="control-row">
          <button className="btn" onClick={() => train(NORMAL_POSTURE_LABEL)} disabled={isTraining}>
            Train Normal Posture
          </button>
          <button className="btn" onClick={() => train(HEAD_LEFT_LABEL)} disabled={isTraining}>
            Train Head Turned Left
          </button>
          <button className="btn" onClick={() => train(HEAD_RIGHT_LABEL)} disabled={isTraining}>
            Train Head Turned Right
          </button>
          <button className="btn" onClick={() => train(STANDING_UP_LABEL)} disabled={isTraining}>
            Train Standing Up
          </button>
          <button className="btn" onClick={() => train(ABSENT_LABEL)} disabled={isTraining}>
            Train Absent
          </button>
          <button className="btn" onClick={saveModel} disabled={isTraining}>
            Save Model
          </button>
          <button className="btn" onClick={loadModel}>
            Load Model
          </button>
          <button className="btn" onClick={run}>Run</button>
        </div>
        */}
      </div>

      {/* Status Section */}
      <div className="status">
        <h2>Current Behavior: {currentBehavior}</h2>
        <h3>Object Detections:</h3>
        <ul>
          {detections.map((detection, index) => (
            <li key={index}>{`${detection.class} (${Math.round(detection.score * 100)}%)`}</li>
          ))}
        </ul>
      </div>
    </div>
  );

}

export default App;
