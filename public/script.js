const socket = io.connect("http://localhost:3030");
socket.on("jediStream", (data) => {
  const bbCanVasObj = document.getElementById("bbCanvas");
  bbCanVasObj.src = `data:image/jpeg;base64,${data}`;
});
socket.on("faceStat", (data) => {
  console.log(data);
});
