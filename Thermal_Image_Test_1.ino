#include <Wire.h>
#include <Adafruit_MLX90640.h>

Adafruit_MLX90640 mlx;
float frame[32*24];

void setup() {
  Serial.begin(115200);
  Wire.begin(21,22);

  if (!mlx.begin(0x33, &Wire)) {
    Serial.println("MLX90640 not detected");
    while (1);
  }

  mlx.setMode(MLX90640_CHESS);
  mlx.setResolution(MLX90640_ADC_18BIT);
  mlx.setRefreshRate(MLX90640_4_HZ);
}

void loop() {

  if (mlx.getFrame(frame) != 0) {
    Serial.println("ERR");
    return;
  }

  for (int i=0; i<768; i++) {
    Serial.print(frame[i]);
    if(i < 767) Serial.print(",");
  }

  Serial.println();
  delay(500);
}