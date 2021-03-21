const cv = require("opencv4nodejs");
const path = require("path");
const express = require("express");
const process = require("process");
const fs = require("file-system");

const cors = require("cors");

// Face-API
const tf = require("@tensorflow/tfjs-node");
const faceapi = require("@vladmandic/face-api");
const { json } = require("express");
const { LabeledFaceDescriptors } = require("@vladmandic/face-api");
let optionsSSDMobileNet;
const faceRecogModels = "./models/";

// Create an instance of express (Start the server)
const app = express();
const server = require("http").Server(app);
const io = require("socket.io")(server, {
  cors: {
    origin: "*",
  },
});
app.use(cors());
app.use(express.json()); // Enables to accept JSON requests.
app.use("/", express.static(path.join(__dirname, "public"))); // Exposed to public.
app.set("view engine", "ejs");

const PORT = 3030;
const HOST_URL = `http://localhost:${PORT}`;
server.listen(PORT);
console.log(`running on ${HOST_URL}`);

const blue = new cv.Vec(255, 0, 0);
const green = new cv.Vec(0, 255, 0);
const red = new cv.Vec(0, 0, 255);

const decode = async (img) => {
  const decoded = tf.node.decodeImage(img);
  const casted = decoded.toFloat();
  const result = casted.expandDims(0);
  decoded.dispose();
  casted.dispose();
  return result;
};

const analyse = async (tensor) => {
  const result = await faceapi
    .detectSingleFace(tensor, optionsSSDMobileNet)
    .withFaceLandmarks()
    .withFaceDescriptor();
  return result;
};

const loadFaceWeights = async () => {
  // Loads trained face features from fileStructure, and parses into json.
  try {
    let faceWeights = [];
    console.log("Reading");
    /* Loading saved face-descriptors */
    const faceWeightsPath = path.join(__dirname, "data/");
    console.log(faceWeightsPath);
    const faceDirObj = fs.opendirSync(faceWeightsPath);
    let filesLeft = true;
    while (filesLeft) {
      // Read a file as fs.Dirent object
      let faceDir = faceDirObj.readSync();

      // If readSync() does not return null
      // print its filename
      if (faceDir != null)
        faceWeights.push(
          JSON.parse(
            fs.readFileSync(path.join(faceWeightsPath, `${faceDir.name}`))
          )
        );
      else filesLeft = false;
    }
    return faceWeights;
  } catch (err) {
    console.log(err);
  }
};

const createDesc = (faceDescriptor) => {
  return new faceapi.LabeledFaceDescriptors("Rashmi Ma'am", faceDescriptor);
};

//Load models from disk.
try {
  faceapi.nets.ssdMobilenetv1
    .loadFromDisk(faceRecogModels)
    .then(faceapi.nets.faceLandmark68Net.loadFromDisk(faceRecogModels))
    .then(faceapi.nets.faceRecognitionNet.loadFromDisk(faceRecogModels))
    .then(
      (optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options({
        minConfidence: 0.1,
        maxResults: 1,
      }))
    )
    .catch((err) => console.error(err));
} catch (err) {
  console.log(err);
}

//Initialize tf.
try {
  faceapi.tf
    .setBackend("tensorflow")
    .then(faceapi.tf.enableProdMode())
    .then(faceapi.tf.ENV.set("DEBUG", false))
    .then(faceapi.tf.ready())
    .then(console.log("Ready!"));
} catch (err) {
  console.log(err);
}

app.get("/", async (req, res) => {
  res.render("bb");
  io.on("connection", (socket) => {
    engageRecog(socket);
    console.log("New user Connected!");

    socket.on("disconnect", () => {
      console.log("User disconnected!");
    });
  });

  //Initialize.
  // try {
  //   await engageRecog(socket);
  // } catch (err) {
  //   console.error(err);
  // }
});

async function engageRecog(socket) {
  //Monitor face-data
  var faceData = { faceStat: null, jediStream: null };
  // Loading prev trained weights, if any.
  const faceWeights = await loadFaceWeights();
  console.log("Face weights Loaded");

  console.log(
    `Version: TensorFlow/JS ${faceapi.tf?.version_core} FaceAPI ${
      faceapi.version.faceapi
    } Backend: ${faceapi.tf?.getBackend()}`
  );

  console.log("ACM's Cloud-Based Facial Recog cognitive API is live ->");
  // OpenCV way to access camStream from front-end.
  const FPS = 30;
  const vCap = new cv.VideoCapture(0);
  vCap.set(cv.CAP_PROP_FRAME_WIDTH, 320);
  vCap.set(cv.CAP_PROP_FRAME_HEIGHT, 480);

  setInterval(async () => {
    const frameDisp = vCap.read();
    const frameDispFlip = frameDisp.flip(1);

    // Tensorflow needs jpeg encoded image.
    const imgJpeg = cv.imencode(".jpeg", frameDispFlip);
    // Convert into tensor.
    const imgDecoded = await decode(imgJpeg);
    // Pass through SSD.
    const imgDesc = await analyse(imgDecoded);
    imgDecoded.dispose();
    if (imgDesc) {
      // Draw bounding rect.
      const x = imgDesc.alignedRect._box._x,
        y = imgDesc.alignedRect._box._y,
        height = imgDesc.alignedRect._box._height,
        width = imgDesc.alignedRect._box._width;
      frameDispFlip.drawRectangle(
        new cv.Point2(x, y),
        new cv.Point2(x + width, y + height),
        green,
        2,
        1,
        0
      );

      /* Create and save labeleDescriptors.*****************************************/
      // const faceDescriptor = createDesc([imgDesc.descriptor]);
      // if (faceDescriptor) {
      //   fs.writeFileSync(
      //     `./data/${Math.random().toString(36).substring(2, 7)}.json`,
      //     JSON.stringify(faceDescriptor.toJSON())
      //   );
      // }

      /* Loads and parse the face weights ******************************************/

      const labels = [
        "Dibyasom",
        "PankajSir",
        "DeepakSir",
        "AnujSir",
        "AmarSir",
        "AvitaMaam",
        "LalitSir",
        "NeeluMaam",
        "PraveenSir",
        "RashmiMaam",
        "RaviSir",
        "AmarendraSir",
        "xR",
      ];
      const labeledFaceWeights = {
        Dibyasom: faceWeights[4],
        PankajSir: faceWeights[7],
        DeepakSir: faceWeights[3],
        AnujSir: faceWeights[1],
        AmarSir: faceWeights[0],
        AvitaMaam: faceWeights[2],
        LalitSir: faceWeights[5],
        NeeluMaam: faceWeights[6],
        PraveenSir: faceWeights[8],
        RashmiMaam: faceWeights[9],
        RaviSir: faceWeights[10],
        AmarendraSir: faceWeights[11],
        xR: faceWeights[12],
      };

      const labeledFaceDescriptors = await Promise.all(
        labels.map(async (label) => {
          return faceapi.LabeledFaceDescriptors.fromJSON(
            labeledFaceWeights[label]
          );
        })
      );

      const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.5);

      // Run face match.
      const bestMatch = faceMatcher.findBestMatch(imgDesc.descriptor);
      console.log("Writing to socket @faceStat-event.");
      socket.emit(("faceStat", "Hello"));
      // socket.emit(("faceStat", bestMatch.toString()));
      console.log(bestMatch);
      faceData.faceStat = bestMatch.toString();
      //Append detected face-coordinates, ****Testing purpose, cus target user is blind anyway.
      const frameJpegEncoded = cv
        .imencode(".jpeg", frameDispFlip)
        .toString("base64");
      faceData.jediStream = frameJpegEncoded;

      /* ***** Dibya daddy assignment karo. ***** */
    } else {
      faceData.faceStat = "No face detected!";
    }

    socket.emit("jediStream", faceData.jediStream);
    socket.emit("faceStat", faceData.faceStat);
  }, 1000 / FPS);
  //
  // socket.emit("hello", "ssup?");
  // GESTURE RECOGNITION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
}

// Aligned rect return format.
/*
  alignedRect: M {
    _imageDims: A { _width: 352, _height: 288 },
    _score: 0.7536973357200623,
    _classScore: 0.7536973357200623,
    _className: '',
    _box: D {
      _x: 164.79627001471817,
      _y: 79.0680174840592,
      _width: 91.74189727753402,
      _height: 89.08994125127792
    }
*/
