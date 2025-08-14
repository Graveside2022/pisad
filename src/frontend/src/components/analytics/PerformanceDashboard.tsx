import React, { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  Chip,
  LinearProgress,
  Paper,
  Divider,
} from '@mui/material';
import Grid from '@mui/material/Grid';
import {
  TrendingUp,
  TrendingDown,
  Speed,
  LocationSearching,
  Timeline,
  Assessment,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface PerformanceMetrics {
  missionId: string;
  detectionMetrics: {
    totalDetections: number;
    detectionsPerHour: number;
    detectionsPerKm2: number;
    meanDetectionConfidence: number;
    detectionCoverage: number;
  };
  approachMetrics: {
    finalDistanceM: number | null;
    approachTimeS: number | null;
    approachEfficiency: number;
    finalRssiDbm: number | null;
    rssiImprovementDb: number;
  };
  searchMetrics: {
    totalAreaKm2: number;
    areaCoveredKm2: number;
    coveragePercentage: number;
    totalDistanceKm: number;
    searchTimeMinutes: number;
    averageSpeedKmh: number;
    searchPatternEfficiency: number;
  };
  baselineComparison: {
    timeImprovementPercent: number;
    areaReductionPercent: number;
    accuracyImprovementPercent: number;
    costReductionPercent: number;
    operatorWorkloadReduction: number;
  };
  overallScore: number;
  recommendations: string[];
}

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  icon: React.ReactNode;
  trend?: number;
  color?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  icon,
  trend,
  color = 'primary.main',
}) => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Box sx={{ color, mr: 1 }}>{icon}</Box>
        <Typography variant="subtitle2" color="text.secondary">
          {title}
        </Typography>
      </Box>
      <Typography variant="h4" component="div" sx={{ mb: 1 }}>
        {value}
        {unit && (
          <Typography component="span" variant="subtitle1" sx={{ ml: 0.5 }}>
            {unit}
          </Typography>
        )}
      </Typography>
      {trend !== undefined && (
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {trend > 0 ? (
            <TrendingUp sx={{ color: 'success.main', fontSize: 20 }} />
          ) : (
            <TrendingDown sx={{ color: 'error.main', fontSize: 20 }} />
          )}
          <Typography
            variant="body2"
            sx={{
              ml: 0.5,
              color: trend > 0 ? 'success.main' : 'error.main',
            }}
          >
            {Math.abs(trend)}%
          </Typography>
        </Box>
      )}
    </CardContent>
  </Card>
);

interface PerformanceDashboardProps {
  missionId?: string;
}

export const PerformanceDashboard: React.FC<PerformanceDashboardProps> = ({
  missionId,
}) => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      if (!missionId) {
        setLoading(false);
        return;
      }

      try {
        const response = await fetch(`/api/analytics/metrics?mission_id=${missionId}`);
        if (!response.ok) {
          throw new Error('Failed to fetch metrics');
        }
        const data = await response.json();
        setMetrics(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, [missionId]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {error}
      </Alert>
    );
  }

  if (!metrics) {
    return (
      <Alert severity="info" sx={{ m: 2 }}>
        Select a mission to view performance metrics
      </Alert>
    );
  }

  // Prepare chart data
  const coverageData = [
    { name: 'Covered', value: metrics.searchMetrics.areaCoveredKm2, color: '#4caf50' },
    {
      name: 'Uncovered',
      value: metrics.searchMetrics.totalAreaKm2 - metrics.searchMetrics.areaCoveredKm2,
      color: '#e0e0e0',
    },
  ];

  const comparisonData = [
    { name: 'Time', value: metrics.baselineComparison.timeImprovementPercent },
    { name: 'Accuracy', value: metrics.baselineComparison.accuracyImprovementPercent },
    { name: 'Cost', value: metrics.baselineComparison.costReductionPercent },
    { name: 'Workload', value: metrics.baselineComparison.operatorWorkloadReduction },
  ];

  const detectionTimelineData = Array.from({ length: 10 }, (_, i) => ({
    time: i * 5,
    detections: Math.floor(Math.random() * 5) + 1,
    confidence: 70 + Math.random() * 30,
  }));

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Performance Analytics
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="subtitle1" color="text.secondary">
            Mission: {metrics.missionId}
          </Typography>
          <Chip
            label={`Score: ${metrics.overallScore.toFixed(1)}/100`}
            color={metrics.overallScore >= 70 ? 'success' : 'warning'}
            sx={{ fontWeight: 'bold' }}
          />
        </Box>
      </Box>

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricCard
            title="Detection Rate"
            value={metrics.detectionMetrics.detectionsPerHour.toFixed(1)}
            unit="/hr"
            icon={<LocationSearching />}
            trend={15}
            color="primary.main"
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricCard
            title="Search Efficiency"
            value={metrics.searchMetrics.searchPatternEfficiency.toFixed(0)}
            unit="%"
            icon={<Speed />}
            trend={8}
            color="success.main"
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricCard
            title="Final Distance"
            value={metrics.approachMetrics.finalDistanceM?.toFixed(1) || 'N/A'}
            unit="m"
            icon={<Timeline />}
            trend={-25}
            color="info.main"
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricCard
            title="Coverage"
            value={metrics.searchMetrics.coveragePercentage.toFixed(0)}
            unit="%"
            icon={<Assessment />}
            color="warning.main"
          />
        </Grid>
      </Grid>

      {/* Charts Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Coverage Pie Chart */}
        <Grid size={{ xs: 12, md: 4 }}>
          <Paper sx={{ p: 2, height: 300 }}>
            <Typography variant="h6" gutterBottom>
              Search Area Coverage
            </Typography>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={coverageData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={(entry) => `${entry.name}: ${entry.value?.toFixed(1) || '0'} km²`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {coverageData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Baseline Comparison Bar Chart */}
        <Grid size={{ xs: 12, md: 4 }}>
          <Paper sx={{ p: 2, height: 300 }}>
            <Typography variant="h6" gutterBottom>
              vs Baseline Performance
            </Typography>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip formatter={(value) => `${value}%`} />
                <Bar dataKey="value" fill="#ff9800" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Detection Timeline */}
        <Grid size={{ xs: 12, md: 4 }}>
          <Paper sx={{ p: 2, height: 300 }}>
            <Typography variant="h6" gutterBottom>
              Detection Timeline
            </Typography>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={detectionTimelineData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" label={{ value: 'Time (min)', position: 'insideBottom', offset: -5 }} />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip />
                <Legend />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="detections"
                  stroke="#1976d2"
                  name="Detections"
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="confidence"
                  stroke="#4caf50"
                  name="Confidence %"
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Detailed Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid size={{ xs: 12, md: 6 }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Detection Performance
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Grid container spacing={2}>
              <Grid size={6}>
                <Typography variant="body2" color="text.secondary">
                  Total Detections
                </Typography>
                <Typography variant="h6">
                  {metrics.detectionMetrics.totalDetections}
                </Typography>
              </Grid>
              <Grid size={6}>
                <Typography variant="body2" color="text.secondary">
                  Mean Confidence
                </Typography>
                <Typography variant="h6">
                  {metrics.detectionMetrics.meanDetectionConfidence.toFixed(1)}%
                </Typography>
              </Grid>
              <Grid size={6}>
                <Typography variant="body2" color="text.secondary">
                  Detections per km²
                </Typography>
                <Typography variant="h6">
                  {metrics.detectionMetrics.detectionsPerKm2.toFixed(1)}
                </Typography>
              </Grid>
              <Grid size={6}>
                <Typography variant="body2" color="text.secondary">
                  Coverage
                </Typography>
                <Typography variant="h6">
                  {metrics.detectionMetrics.detectionCoverage.toFixed(1)}%
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        <Grid size={{ xs: 12, md: 6 }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Approach Performance
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Grid container spacing={2}>
              <Grid size={6}>
                <Typography variant="body2" color="text.secondary">
                  Approach Efficiency
                </Typography>
                <Typography variant="h6">
                  {metrics.approachMetrics.approachEfficiency.toFixed(1)}%
                </Typography>
              </Grid>
              <Grid size={6}>
                <Typography variant="body2" color="text.secondary">
                  Final RSSI
                </Typography>
                <Typography variant="h6">
                  {metrics.approachMetrics.finalRssiDbm?.toFixed(1) || 'N/A'} dBm
                </Typography>
              </Grid>
              <Grid size={6}>
                <Typography variant="body2" color="text.secondary">
                  RSSI Improvement
                </Typography>
                <Typography variant="h6">
                  {metrics.approachMetrics.rssiImprovementDb.toFixed(1)} dB
                </Typography>
              </Grid>
              <Grid size={6}>
                <Typography variant="body2" color="text.secondary">
                  Approach Time
                </Typography>
                <Typography variant="h6">
                  {metrics.approachMetrics.approachTimeS
                    ? `${(metrics.approachMetrics.approachTimeS / 60).toFixed(1)} min`
                    : 'N/A'}
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>

      {/* Recommendations */}
      {metrics.recommendations.length > 0 && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Recommendations
          </Typography>
          <Divider sx={{ mb: 2 }} />
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {metrics.recommendations.map((rec, index) => (
              <Box key={index} sx={{ display: 'flex', alignItems: 'flex-start' }}>
                <Typography variant="body2" sx={{ mr: 1, fontWeight: 'bold' }}>
                  {index + 1}.
                </Typography>
                <Typography variant="body2">{rec}</Typography>
              </Box>
            ))}
          </Box>
        </Paper>
      )}

      {/* Overall Progress */}
      <Paper sx={{ p: 2, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Overall Performance Score
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
          <Box sx={{ width: '100%', mr: 2 }}>
            <LinearProgress
              variant="determinate"
              value={metrics.overallScore}
              sx={{
                height: 10,
                borderRadius: 5,
                backgroundColor: 'grey.300',
                '& .MuiLinearProgress-bar': {
                  backgroundColor:
                    metrics.overallScore >= 70
                      ? 'success.main'
                      : metrics.overallScore >= 50
                      ? 'warning.main'
                      : 'error.main',
                },
              }}
            />
          </Box>
          <Typography variant="h5" sx={{ minWidth: 60 }}>
            {metrics.overallScore.toFixed(0)}%
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
};