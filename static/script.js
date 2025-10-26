async function getRecommendation() {
  const movieName = document.getElementById("movieInput").value;
  const topN = parseInt(document.getElementById("topNInput").value);
  if (!movieName) {
    alert("Please enter a movie name!");
    return;
  }
  if (!topN || topN < 1) {
    alert("Please enter a valid number for top_n!");
    return;
  }

  const response = await fetch("http://127.0.0.1:8000/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ description: movieName, top_n: topN }),
  });

  if (!response.ok) {
    alert("Error: " + response.status);
    return;
  }
  const data = await response.json();
  const movies = data.recommendations;
  let html = "<h2>Recommended Movies</h2><ul>";
  movies.forEach((movie) => {
    html += `<li>${movie}</li>`;
  });
  html += "</ul>";

  document.getElementById("result").innerHTML = html;
}
