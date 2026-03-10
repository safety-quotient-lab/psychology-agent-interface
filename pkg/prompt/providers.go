package prompt

// IdentityProvider returns the persona prompt for a given model tier.
type IdentityProvider interface {
	Name() string
	Prompt(tier int) string
}

// ToolProvider returns tool descriptions for the system prompt.
type ToolProvider interface {
	ToolList() string         // "shell, read_file, ..." — for native mode
	ToolDescriptions() string // full descriptions — for ReAct mode
}

// ContextProvider returns workspace context.
type ContextProvider interface {
	WorkingDir() string
	FileList() string
	ProjectInstructions() string
}

// FormatProvider returns format rules and reminders for a model tier.
type FormatProvider interface {
	Rules(tier int) string // appended to system prompt tail
	Nudge(tier int) string // per-turn reinforcement
}

// FewShotProvider returns example exchanges for a model tier.
type FewShotProvider interface {
	Examples(tier int) []Message
}
