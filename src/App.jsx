import { CircleMarker, MapContainer, Popup, TileLayer } from 'react-leaflet'
import './App.css'

const mapCenter = [31.7917, -7.0926]

// Use OpenStreetMap tiles by default. If you prefer CARTO, switch the URL below.
const tileUrl = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'

const spotlightCities = [
  {
    name: 'Rabat',
    description: 'Administrative capital and gateway to the Atlantic',
    coords: [34.020882, -6.84165],
    color: '#f97316',
  },
  {
    name: 'Casablanca',
    description: 'Largest Atlantic port and industrial hub',
    coords: [33.57311, -7.58984],
    color: '#22d3ee',
  },
  {
    name: 'Marrakesh',
    description: 'Historic inland marketplace and tourism magnet',
    coords: [31.62947, -7.98108],
    color: '#a855f7',
  },
  {
    name: 'Agadir',
    description: 'Key agricultural export corridor in Souss-Massa',
    coords: [30.427755, -9.598107],
    color: '#84cc16',
  },
]

function App() {
  return (
    <div className="app-shell">
      <header className="headline">
        <p className="eyebrow">Morocco focus</p>
        <h1>Fertilizer Recon Morocco Map</h1>
        <p className="lede">
          A lightweight base map centered on the Kingdom of Morocco to kick off
          our spatial exploration work. Hover or tap the highlighted regions to
          learn more about the primary coastal and inland hubs.
        </p>
      </header>

      <section className="map-panel">
        <div className="map-wrapper">
          <MapContainer
            center={mapCenter}
            zoom={5}
            minZoom={4}
            maxZoom={8}
            scrollWheelZoom
            className="morocco-map"
          >
            <TileLayer
              url={tileUrl}
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              eventHandlers={{
                tileerror: (e) => {
                  // Log tile loading errors to the browser console for debugging
                  // If tiles fail, open the network tab and look for requests to the tile host.
                  // The user-facing map markers will still render as they are vector elements.
                  // eslint-disable-next-line no-console
                  console.warn('Leaflet tile error', e)
                },
              }}
            />

            {spotlightCities.map((city) => (
              <CircleMarker
                key={city.name}
                center={city.coords}
                radius={10}
                pathOptions={{
                  color: city.color,
                  fillColor: city.color,
                  fillOpacity: 0.6,
                  weight: 2,
                }}
              >
                <Popup>
                  <strong>{city.name}</strong>
                  <br />
                  {city.description}
                </Popup>
              </CircleMarker>
            ))}
          </MapContainer>
        </div>

        <ul className="city-list">
          {spotlightCities.map((city) => (
            <li key={city.name} className="city-card">
              <span
                className="swatch"
                style={{ backgroundColor: city.color }}
                aria-hidden="true"
              />
              <div>
                <p className="city-name">{city.name}</p>
                <p className="city-description">{city.description}</p>
              </div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}

export default App
