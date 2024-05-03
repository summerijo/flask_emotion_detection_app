function toggleSidebar() {
    var sidebar = document.getElementById("mySidebar");
    sidebar.classList.toggle("closed");
    var content = document.querySelector(".content");
    content.classList.toggle("closed");
    var navbar = document.querySelector(".navbar");
    navbar.classList.toggle("closed");
}

var realTimeEnabled = false;

function startRealTime() {
    realTimeEnabled = true;
    document.getElementById("emotion_video").src = "/real_time";
    document.getElementById("real_time_button").innerText = "Stop Real-time Emotion Detection";
}

function stopRealTime() {
    realTimeEnabled = false;
    document.getElementById("emotion_video").src = "";
    document.getElementById("real_time_button").innerText = "Start Real-time Emotion Detection";
}

function uploadImage() {
    stopRealTime(); // Stop real-time detection before uploading image
    var fileInput = document.getElementById("file_input");
    if (fileInput.files.length === 0) {
        alert("Please select a file.");
        return;
    }
    document.getElementById("upload_form").submit();
}

