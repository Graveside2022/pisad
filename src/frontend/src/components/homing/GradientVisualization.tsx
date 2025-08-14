/**
 * Gradient Visualization Widget
 * Shows a visual representation of the RSSI gradient field
 */

import React from "react";
import { Box, Card, CardContent, Typography, useTheme } from "@mui/material";
import { useWebSocket } from "../../hooks/useWebSocket";

interface GradientData {
  gradient_direction: number;
  gradient_magnitude: number;
  gradient_confidence: number;
  rssi_samples: Array<{
    x: number;
    y: number;
    rssi: number;
  }>;
}

const GradientVisualization: React.FC = () => {
  const theme = useTheme();
  const { addMessageHandler } = useWebSocket();
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [gradientData, setGradientData] = React.useState<GradientData | null>(
    null,
  );

  // Update gradient data from WebSocket
  React.useEffect(() => {
    const handleMessage = (message: any) => {
      if (message?.type === "homing_gradient") {
        setGradientData(message.data);
      }
    };

    const unsubscribe = addMessageHandler(handleMessage);
    return () => unsubscribe();
  }, [addMessageHandler]);

  // Draw gradient visualization
  React.useEffect(() => {
    if (!canvasRef.current || !gradientData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Set up coordinate system (center origin)
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const scale = 10; // pixels per meter

    // Draw grid
    ctx.strokeStyle = theme.palette.divider;
    ctx.lineWidth = 0.5;
    ctx.setLineDash([5, 5]);

    // Vertical lines
    for (let x = 0; x < canvas.width; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }

    // Horizontal lines
    for (let y = 0; y < canvas.height; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    ctx.setLineDash([]);

    // Draw RSSI samples as heat map points
    if (gradientData.rssi_samples) {
      gradientData.rssi_samples.forEach((sample) => {
        const x = centerX + sample.x * scale;
        const y = centerY - sample.y * scale; // Flip Y axis

        // Map RSSI to color (red = weak, green = strong)
        const normalizedRssi = (sample.rssi + 100) / 70; // Normalize -100 to -30 dBm
        const hue = normalizedRssi * 120; // 0 (red) to 120 (green)

        ctx.fillStyle = `hsla(${hue}, 70%, 50%, 0.6)`;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.fill();

        // Draw RSSI value
        ctx.fillStyle = theme.palette.text.primary;
        ctx.font = "10px monospace";
        ctx.textAlign = "center";
        ctx.fillText(`${sample.rssi.toFixed(0)}`, x, y + 20);
      });
    }

    // Draw drone position (center)
    ctx.fillStyle = theme.palette.primary.main;
    ctx.strokeStyle = theme.palette.primary.dark;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(centerX, centerY, 10, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Draw gradient arrow
    if (gradientData.gradient_magnitude > 0) {
      const arrowLength = Math.min(gradientData.gradient_magnitude * 500, 100);
      const angle = (gradientData.gradient_direction * Math.PI) / 180;
      const endX = centerX + Math.cos(angle) * arrowLength;
      const endY = centerY - Math.sin(angle) * arrowLength; // Flip Y axis

      // Arrow shaft
      ctx.strokeStyle = theme.palette.success.main;
      ctx.lineWidth = 3;
      ctx.globalAlpha = gradientData.gradient_confidence / 100;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(endX, endY);
      ctx.stroke();

      // Arrow head
      const headLength = 15;
      const headAngle = Math.PI / 6;
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(
        endX - headLength * Math.cos(angle - headAngle),
        endY + headLength * Math.sin(angle - headAngle),
      );
      ctx.moveTo(endX, endY);
      ctx.lineTo(
        endX - headLength * Math.cos(angle + headAngle),
        endY + headLength * Math.sin(angle + headAngle),
      );
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Draw compass
    ctx.strokeStyle = theme.palette.text.secondary;
    ctx.fillStyle = theme.palette.text.secondary;
    ctx.lineWidth = 1;
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";

    // North
    ctx.beginPath();
    ctx.moveTo(centerX, 20);
    ctx.lineTo(centerX, 35);
    ctx.stroke();
    ctx.fillText("N", centerX, 15);

    // East
    ctx.beginPath();
    ctx.moveTo(canvas.width - 20, centerY);
    ctx.lineTo(canvas.width - 35, centerY);
    ctx.stroke();
    ctx.fillText("E", canvas.width - 10, centerY + 4);

    // South
    ctx.beginPath();
    ctx.moveTo(centerX, canvas.height - 20);
    ctx.lineTo(centerX, canvas.height - 35);
    ctx.stroke();
    ctx.fillText("S", centerX, canvas.height - 5);

    // West
    ctx.beginPath();
    ctx.moveTo(20, centerY);
    ctx.lineTo(35, centerY);
    ctx.stroke();
    ctx.fillText("W", 10, centerY + 4);

    // Draw scale
    ctx.strokeStyle = theme.palette.text.primary;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(20, canvas.height - 30);
    ctx.lineTo(20 + scale * 5, canvas.height - 30);
    ctx.stroke();

    // Scale ticks
    ctx.beginPath();
    ctx.moveTo(20, canvas.height - 25);
    ctx.lineTo(20, canvas.height - 35);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(20 + scale * 5, canvas.height - 25);
    ctx.lineTo(20 + scale * 5, canvas.height - 35);
    ctx.stroke();

    ctx.fillStyle = theme.palette.text.primary;
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("5m", 20 + (scale * 5) / 2, canvas.height - 15);
  }, [gradientData, theme]);

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Gradient Field Visualization
        </Typography>

        <Box position="relative">
          <canvas
            ref={canvasRef}
            width={400}
            height={400}
            style={{
              width: "100%",
              maxWidth: 400,
              height: "auto",
              aspectRatio: "1",
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: theme.shape.borderRadius,
            }}
          />

          {gradientData && (
            <Box
              position="absolute"
              top={8}
              left={8}
              bgcolor="background.paper"
              p={1}
              borderRadius={1}
            >
              <Typography variant="caption" display="block">
                Gradient: {gradientData.gradient_magnitude.toFixed(3)} dB/m
              </Typography>
              <Typography variant="caption" display="block">
                Direction: {gradientData.gradient_direction.toFixed(1)}°
              </Typography>
              <Typography variant="caption" display="block">
                Confidence: {gradientData.gradient_confidence.toFixed(1)}%
              </Typography>
            </Box>
          )}
        </Box>

        <Box mt={2}>
          <Typography variant="caption" color="textSecondary">
            • Circle size indicates RSSI sample location
          </Typography>
          <br />
          <Typography variant="caption" color="textSecondary">
            • Color: Red (weak) to Green (strong) signal
          </Typography>
          <br />
          <Typography variant="caption" color="textSecondary">
            • Arrow shows gradient direction and magnitude
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default GradientVisualization;
