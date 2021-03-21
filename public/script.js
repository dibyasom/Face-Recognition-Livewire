// const socket = io.connect("http://localhost:3030");
// const bbCanVasObj = document.getElementById("bbCanvas");

// var socket = io();

// socket.on("connect", () => {
//   console.log("Connected ->", socket.id);
// });

// const videoCap = () => {
//   navigator.getUsermedia(
//     { video: {} },
//     (stream) => {
//       bbCanVasObj.srcObject = stream;
//       console.log(stream);
//     },
//     (err) => console.err(err)
//   );
// };

const socket = io.connect("http://localhost:3030");
const resultBoard = document.getElementById("result");

// Live Stream
socket.on("jediStream", (data) => {
  const bbCanVasObj = document.getElementById("bbCanvas");
  bbCanVasObj.src = `data:image/jpeg;base64,${data}`;
});

// Face scan result.
socket.on("faceStat", (data) => {
  resultBoard.innerHTML = data;
});
