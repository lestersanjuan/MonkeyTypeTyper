function sendData() {
  const input = document.getElementById("inputText").value;

  fetch("/api/process", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text: input }),
  })
    .then((res) => res.json())
    .then((data) => {
      document.getElementById("output").innerText = data.result;
    });
}
