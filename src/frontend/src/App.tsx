import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { Container, Grid, AppBar, Toolbar, Typography } from "@mui/material";
import theme from "./theme";
import Dashboard from "./components/dashboard/Dashboard";

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Grid container direction="column" sx={{ minHeight: "100vh" }}>
          <Grid>
            <AppBar position="static" elevation={0}>
              <Toolbar>
                <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                  PISAD - RF-Homing SAR Drone System
                </Typography>
              </Toolbar>
            </AppBar>
          </Grid>

          <Grid sx={{ flexGrow: 1 }}>
            <Container maxWidth={false} sx={{ py: 3 }}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
              </Routes>
            </Container>
          </Grid>
        </Grid>
      </Router>
    </ThemeProvider>
  );
}

export default App;
