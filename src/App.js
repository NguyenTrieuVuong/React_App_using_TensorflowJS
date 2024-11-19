import * as tf from '@tensorflow/tfjs';
import React, { useEffect, useRef, useState } from 'react';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import * as cocossd from '@tensorflow-models/coco-ssd';
import { initNotifications, notify } from '@mycv/f8-notification';
import Webcam from "react-webcam";
import './App.css';

tf.setBackend('webgl');

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

  const init = async () => {
    console.log('Initializing...');
    await setupCamera();

    console.log('Camera setup success.');

    classifier.current = knnClassifier.create();
    mobilenetModule.current = await mobilenet.load();
    cocoModel.current = await cocossd.load(); // Load COCO-SSD model

    console.log('Setup complete.');
    console.log('Train each behavior using the buttons below.');

    initNotifications({ cooldown: 3000 });

    // Start COCO-SSD object detection
    runObjectDetection();
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

  const train = async label => {
    console.log(`[ ${label} ] is training...`);
    for (let i = 0; i < TRAINING_TIMES; ++i) {
      console.log(`Progress ${(i + 1) / TRAINING_TIMES * 100}%`);
      await training(label);
    }
    console.log(`[ ${label} ] training complete.`);
  };

  const training = label => {
    return new Promise(async resolve => {
      const embedding = mobilenetModule.current.infer(
        video.current,
        true
      );
      classifier.current.addExample(embedding, label);
      await sleep(100);
      resolve();
    });
  };

  const run = async () => {
    const embedding = mobilenetModule.current.infer(
      video.current,
      true
    );
    const result = await classifier.current.predictClass(embedding);

    if (result.confidences[result.label] > DETECTION_CONFIDENCE) {
      setCurrentBehavior(result.label);

      switch (result.label) {
        case NORMAL_POSTURE_LABEL:
          console.log('Normal posture detected');
          break;
        case HEAD_LEFT_LABEL:
          console.log('Head turned left detected');
          break;
        case HEAD_RIGHT_LABEL:
          console.log('Head turned right detected');
          break;
        case STANDING_UP_LABEL:
          console.log('Standing up detected');
          break;
        case ABSENT_LABEL:
          console.log('Absent detected');
          break;
        default:
          console.log('Unknown behavior detected');
      }
    }

    await sleep(200); // Delay between predictions
    run();
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

  const sleep = (ms = 0) => {
    return new Promise(resolve => setTimeout(resolve, ms));
  };

  useEffect(() => {
    init();

    // Cleanup
    return () => {};
  }, []);

  return (
    <div className={`main ${currentBehavior}`}>
      <video
        ref={video}
        className="video"
        autoPlay
      />
      <canvas
        ref={canvasRef}
        className="canvas"
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          zIndex: 8,
        }}
      />
      <div className="control">
        <button className='btn' onClick={() => train(NORMAL_POSTURE_LABEL)}>Train Normal Posture</button>
        <button className='btn' onClick={() => train(HEAD_LEFT_LABEL)}>Train Head Turned Left</button>
        <button className='btn' onClick={() => train(HEAD_RIGHT_LABEL)}>Train Head Turned Right</button>
        <button className='btn' onClick={() => train(STANDING_UP_LABEL)}>Train Standing Up</button>
        <button className='btn' onClick={() => train(ABSENT_LABEL)}>Train Absent</button>
        <button className='btn' onClick={() => run()}>Run</button>
      </div>

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
