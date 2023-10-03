$(function () {
  generated_chart = "";
  let legendWithinBarsPlugin = "";
  $(document).on("click", ".fa-info-circle", function () {
    if (generated_chart) {
      //   var charts = Chart.instances;

      for (const key in Chart.instances) {
        Chart.instances[key].destroy();
      }
    }
    registerChartPlugin();

    createDoughnutChart($(this), "video-report");
    createDoughnutChart($(this), "text-report");
  });
});

function createDoughnutChart(target, targetClass) {
  const reportData = JSON.parse(target.data(targetClass).replaceAll("'", '"'));
  const chartData = getChartData(reportData);

  if (targetClass == "video-report") {
    chartTarget = "video_report_canvas";
  } else if (targetClass == "text-report") {
    chartTarget = "text_report_canvas";
  }

  generated_chart = new Chart(chartTarget, {
    type: "doughnut",
    data: chartData,
    options: getChartOptions(),
  });
}

function getChartData(chartData) {
  const chartValuesLabels = generateLabels(chartData);

  return {
    labels: chartValuesLabels[0],
    datasets: [
      {
        data: chartValuesLabels[1],
        backgroundColor: [
          "#373742",
          "#E6E6ED",
          "#EA1B3D",
          "#676775",
          "#EB4C5E",
        ],
        hoverOffset: 4,
      },
    ],
  };
}

function generateLabels(data) {
  const labels = [];
  const values = [];

  const desiredOrder = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
  ];

  const orderedTextData = {};

  for (const key of desiredOrder) {
    if (data.hasOwnProperty(key)) {
      orderedTextData[key] = data[key];
    }
  }

  for (const key in orderedTextData) {
    labels.push(key.charAt(0).toUpperCase() + key.slice(1));
    values.push(Number(orderedTextData[key]));
  }

  return [labels, values];
}

function registerChartPlugin() {
  legendWithinBarsPlugin = {
    afterDraw: (chart) => {
      console.log("in");
      if (!chart.options.plugins.legendWithinBars) {
        return;
      }
      console.log("in");

      const ctx = chart.ctx;
      const dataset = chart.data.datasets[0];

      for (let i = 0; i < dataset.data.length; i++) {
        const x = dataset.data[i].x;
        const y = dataset.data[i].y;

        // Adjust the legend position within the bars
        const legendX = chart.scales.x.getPixelForValue(x);
        const legendY = chart.scales.y.getPixelForValue(y);

        // Customize the legend appearance (e.g., color, size)
        ctx.fillStyle = "blue"; // Set the legend color
        ctx.fillRect(legendX, legendY - 10, 30, 10); // Customize the legend box size

        // Add text label next to the legend box
        ctx.fillStyle = "black"; // Set the text color
        ctx.fillText("Label " + (i + 1), legendX + 40, legendY + 4); // Customize text positioning
      }
    },
  };
}

function getChartOptions() {
  return {
    plugins: {
      tooltip: {
        callbacks: {
          title: function () {
            return ""; // Return an empty string to hide the title label
          },
          label: function (context) {
            const label = context.label || "";
            const percentage =
              (context.raw / context.dataset.data.reduce((a, b) => a + b)) *
              100;

            return ` ${label}: ${percentage.toFixed(2)}%`;
          },
        },
      },
      legend: {
        display: true,
        position: "bottom",
      },
      // legendWithinBars: true, // Enable the custom plugin
    },
    responsive: true, // Make the chart responsive
  };
}