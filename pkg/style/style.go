// Package style provides shared Lip Gloss styles for the cc CLI.
// Colors match the oksolar dark palette from kashifshah.net.
package style

import (
	"fmt"

	"github.com/charmbracelet/lipgloss"
)

var (
	// oksolar dark palette — https://kashifshah.net/assets/css/main.css
	Purple     = lipgloss.Color("#7d80d1") // violet
	PurpleDim  = lipgloss.Color("#259d94") // cyan
	Green      = lipgloss.Color("#819500")
	Yellow     = lipgloss.Color("#d56500") // orange (site brand)
	Red        = lipgloss.Color("#f23749")
	White      = lipgloss.Color("#98a8a8") // fg-primary
	Gray       = lipgloss.Color("#5b7279") // fg-secondary
	DimGray    = lipgloss.Color("#3d5562") // mid-tone between bg-highlight and fg-secondary

	// Text styles
	Title = lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#093946")).
		Background(Purple).
		Padding(0, 1)

	Heading = lipgloss.NewStyle().
		Bold(true).
		Foreground(Yellow)

	Label = lipgloss.NewStyle().
		Foreground(Gray).
		Width(14)

	Value = lipgloss.NewStyle().
		Foreground(White)

	Dim = lipgloss.NewStyle().
		Foreground(DimGray)

	Success = lipgloss.NewStyle().
		Foreground(Green)

	Warning = lipgloss.NewStyle().
		Foreground(Yellow)

	Error = lipgloss.NewStyle().
		Foreground(Red).
		Bold(true)

	// Box for panels
	Box = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(PurpleDim).
		Padding(1, 2)

	// Command display
	Command = lipgloss.NewStyle().
		Foreground(Green).
		Bold(true)
)

// ColorPct returns a colored percentage string based on thresholds.
func ColorPct(pct int) string {
	s := lipgloss.NewStyle()
	switch {
	case pct > 85:
		return s.Foreground(Red).Render(fmt.Sprintf("%d%%", pct))
	case pct > 70:
		return s.Foreground(Yellow).Render(fmt.Sprintf("%d%%", pct))
	default:
		return s.Foreground(Green).Render(fmt.Sprintf("%d%%", pct))
	}
}
