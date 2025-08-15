/**
 * Centralized Icon Exports
 * Story 4.9 Task 11: Frontend Bundle Optimization
 *
 * This barrel file provides optimized icon imports to enable proper tree-shaking.
 * Instead of importing from @mui/icons-material directly, components should import
 * from this file to ensure only used icons are included in the bundle.
 */

// Status Icons - Used across multiple components
export {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon
} from "@mui/icons-material/";

// Control Icons - Used in control panels
export {
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  SkipNext as SkipNextIcon,
  SkipPrevious as SkipPreviousIcon,
  Replay as ReplayIcon
} from "@mui/icons-material/";

// Configuration Icons
export {
  Settings as SettingsIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Add as AddIcon
} from "@mui/icons-material/";

// Navigation Icons
export {
  LocationOn as LocationOnIcon,
  Navigation as NavigationIcon,
  Map as MapIcon,
  MyLocation as MyLocationIcon,
  GpsFixed as GpsFixedIcon,
  GpsNotFixed as GpsNotFixedIcon,
  GpsOff as GpsOffIcon
} from "@mui/icons-material/";

// Signal/Communication Icons
export {
  SignalCellularAlt as SignalCellularAltIcon,
  NetworkCheck as NetworkCheckIcon,
  WifiTethering as WifiTetheringIcon,
  CellTower as CellTowerIcon
} from "@mui/icons-material/";

// Flight/Drone Icons
export {
  FlightTakeoff as FlightTakeoffIcon,
  FlightLand as FlightLandIcon,
  Flight as FlightIcon,
  Speed as SpeedIcon,
  Timer as TimerIcon
} from "@mui/icons-material/";

// Safety/Emergency Icons
export {
  Emergency as EmergencyIcon,
  StopCircle as StopCircleIcon,
  ReportProblem as ReportProblemIcon,
  Security as SecurityIcon,
  Shield as ShieldIcon
} from "@mui/icons-material/";

// Data/Analytics Icons
export {
  Assessment as AssessmentIcon,
  Analytics as AnalyticsIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Timeline as TimelineIcon,
  ShowChart as ShowChartIcon
} from "@mui/icons-material/";

// File/Download Icons
export {
  GetApp as GetAppIcon,
  CloudDownload as CloudDownloadIcon,
  FileDownload as FileDownloadIcon,
  Description as DescriptionIcon
} from "@mui/icons-material/";

// Battery Icons
export {
  BatteryFull as BatteryFullIcon,
  Battery90 as Battery90Icon,
  Battery60 as Battery60Icon,
  Battery30 as Battery30Icon,
  Battery20 as Battery20Icon,
  BatteryAlert as BatteryAlertIcon,
  BatteryUnknown as BatteryUnknownIcon
} from "@mui/icons-material/";

// Visibility Icons
export {
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon
} from "@mui/icons-material/";
