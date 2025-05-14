const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const summaryDiv = document.getElementById("summary");

<<<<<<< HEAD
const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

let constraints = {
    video: {
        facingMode: isMobile ? 'environment' : 'user'
    }
};


navigator.mediaDevices.getUserMedia(constraints)
=======
navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
>>>>>>> 20e3d8e0caf71a7ff5404b6b86b27ef042d60d12
    .then(stream => video.srcObject = stream)
    .catch(err => alert("Error accessing camera: " + err));

function capture() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob(blob => {
    const formData = new FormData();
    formData.append("image", blob, "capture.jpg");

    fetch("/process", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        summaryDiv.textContent = "Summary:\n" + data.summary;
    })
    .catch(err => alert("Error: " + err));
    }, "image/jpeg");
}