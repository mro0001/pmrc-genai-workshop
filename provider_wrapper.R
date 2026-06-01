# provider_wrapper.R ------------------------------------------------------------
# One helper, `new_chat()`, so workshop code is not locked to a single LLM vendor.
#
# It returns an ellmer chat object for any supported provider, so every downstream
# call (`$chat()`, `$chat_structured()`, schemas built with `type_*`) stays IDENTICAL
# regardless of which provider you pick. Switching providers is a one-word change.
#
#   source("provider_wrapper.R")
#   chat <- new_chat("claude", system_prompt = "You classify tweets.")
#   chat$chat_structured("TWEET: ...", type = my_schema)
#
# Providers (v2 — extended for institutional backends):
#   Direct vendor API key ...... openai, claude, gemini
#   OpenAI-compatible endpoint .. mindrouter, gateway   (any campus gateway that issues a key)
#   Institutional cloud ........ azure   (Azure OpenAI, behind a campus Microsoft agreement)
#                                vertex  (Google Vertex AI, behind a campus Google agreement)
#                                bedrock (AWS Bedrock, serves Claude/Llama/etc.)
#   Local open-weight .......... ollama  (no key, fully private; IRB/FERPA-friendly)
#
# WHY THE BRANCHES DIFFER: not every backend authenticates with a simple API key.
#   - openai/claude/gemini/mindrouter/gateway/azure -> API key (an env var)
#   - vertex  -> Google Application Default Credentials (`gcloud auth application-default login`)
#                plus a project id + location; there is NO api key.
#   - bedrock -> the AWS credential chain (an AWS profile or AWS_* env vars); NO api key.
#   - ollama  -> nothing; it talks to a local server (default http://localhost:11434).
#
# Requires: ellmer (>= 0.4.0). install.packages("ellmer")
# --------------------------------------------------------------------------------

suppressMessages(library(ellmer))

# small null-coalescing helper
`%||%` <- function(a, b) if (is.null(a)) b else a

# Return the first non-empty environment variable from `vars`, or "" if none set.
.first_env <- function(vars) {
  for (v in vars) {
    val <- Sys.getenv(v)
    if (nzchar(val)) return(val)
  }
  ""
}

# Each provider declares: a label, a default model, an auth style, and the
# env var(s)/config it needs. `auth` is one of:
#   "key"        -> resolve an API key from `keys` (first non-empty wins)
#   "google_adc" -> Vertex AI: project + location env vars; auth via gcloud ADC
#   "aws"        -> Bedrock: AWS credential chain (optional AWS_PROFILE)
#   "none"       -> local Ollama; no credentials
.PROVIDERS <- list(
  openai = list(
    label = "OpenAI", default = "gpt-4o-mini", auth = "key",
    keys = c("OPENAI_API_KEY")
  ),
  claude = list(
    label = "Claude (Anthropic)", default = NULL, auth = "key",  # NULL => ellmer default
    keys = c("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
  ),
  gemini = list(
    label = "Gemini (Google AI Studio)", default = "gemini-2.0-flash", auth = "key",
    keys = c("GEMINI_API_KEY", "GOOGLE_API_KEY")
  ),
  mindrouter = list(
    label = "MindRouter (U of Idaho)", default = "qwen2.5:72b", auth = "key",
    base_url = "https://mindrouter.uidaho.edu/v1",
    keys = c("MINDROUTER2_KEY", "MINDROUTER_KEY")
  ),

  # --- institutional self-hosted OpenAI-compatible gateway --------------------
  # e.g. Harvard HUIT AI gateway, UMBC Amplify, UT Dallas CometAI, GMU PatriotAI,
  # CityUHK Chatbot, a campus DeepSeek deployment. Point it at the gateway's
  # OpenAI-compatible base URL (usually ends in /v1) and supply the campus key.
  gateway = list(
    label = "Institutional OpenAI-compatible gateway", default = NULL, auth = "key",
    base_url_env = "CAMPUS_AI_BASE_URL",        # e.g. https://ai-gateway.your.edu/v1
    keys = c("CAMPUS_AI_KEY")
  ),

  # --- institution's Azure OpenAI Service (behind the campus Microsoft tenant) -
  # The model name IS the Azure *deployment* name (deployment_id is deprecated).
  azure = list(
    label = "Azure OpenAI (institutional)", default = NULL, auth = "key",
    endpoint_env = "AZURE_OPENAI_ENDPOINT",     # https://<resource>.openai.azure.com
    api_version  = "2024-10-21",
    keys = c("AZURE_OPENAI_API_KEY")
  ),

  # --- institution's Google Vertex AI (behind the campus Google Cloud project) -
  # No API key: authenticate once with `gcloud auth application-default login`.
  vertex = list(
    label = "Google Vertex AI (institutional)", default = "gemini-2.0-flash", auth = "google_adc",
    project_env  = c("VERTEX_PROJECT", "GOOGLE_CLOUD_PROJECT"),
    location_env = c("VERTEX_LOCATION", "GOOGLE_CLOUD_LOCATION")
  ),

  # --- institution's AWS Bedrock (serves Claude, Llama, etc.) ------------------
  # No API key: uses the AWS credential chain (set AWS_PROFILE or AWS_* env vars).
  bedrock = list(
    label = "AWS Bedrock (institutional)", default = NULL, auth = "aws",
    profile_env = "AWS_PROFILE"
  ),

  # --- local open-weight model via Ollama -------------------------------------
  # No key, no egress. Install Ollama, `ollama pull <model>`, then use it here.
  ollama = list(
    label = "Ollama (local open-weight)", default = "llama3.1", auth = "none",
    base_url_env = "OLLAMA_BASE_URL"            # default http://localhost:11434
  )
)

#' Open a chat with any supported provider.
#'
#' @param provider One of names(.PROVIDERS): "openai", "claude", "gemini",
#'   "mindrouter", "gateway", "azure", "vertex", "bedrock", "ollama".
#' @param model    Model name. NULL uses a sensible default for the provider.
#'   For azure this must be your *deployment* name; for bedrock a Bedrock model id.
#' @param system_prompt Optional system prompt (string).
#' @param base_url Optional override for "gateway"/"ollama" base URL.
#' @param ...      Passed through to the underlying ellmer constructor.
#' @return An ellmer Chat object (use $chat() and $chat_structured() as usual).
new_chat <- function(provider = names(.PROVIDERS),
                     model = NULL,
                     system_prompt = NULL,
                     base_url = NULL,
                     ...) {
  provider <- match.arg(provider)
  cfg <- .PROVIDERS[[provider]]
  if (is.null(model)) model <- cfg$default      # may stay NULL -> ellmer default

  # For key-auth providers, resolve the key now and fail early with a clear message.
  key <- ""
  if (identical(cfg$auth, "key")) {
    key <- .first_env(cfg$keys)
    if (!nzchar(key)) {
      stop(sprintf(
        "No API key found for %s. Set it first, e.g.:\n  Sys.setenv(%s = \"your-key\")",
        cfg$label, cfg$keys[1]), call. = FALSE)
    }
  }

  switch(provider,
    # `credentials` (a function returning the key) is ellmer's current contract;
    # the bare `api_key=` argument is deprecated as of ellmer 0.4.0.
    openai = chat_openai(
      model = model, system_prompt = system_prompt, credentials = function() key, ...
    ),
    claude = chat_anthropic(
      model = model, system_prompt = system_prompt, credentials = function() key, ...
    ),
    gemini = chat_google_gemini(
      model = model, system_prompt = system_prompt, credentials = function() key, ...
    ),
    # MindRouter is an OpenAI-compatible endpoint.
    mindrouter = chat_openai_compatible(
      base_url = cfg$base_url, name = "MindRouter",
      model = model, system_prompt = system_prompt, credentials = function() key, ...
    ),

    # Any institutional OpenAI-compatible gateway (same mechanism as MindRouter).
    gateway = {
      url <- base_url %||% Sys.getenv(cfg$base_url_env)
      if (!nzchar(url)) stop(sprintf(
        "No gateway URL. Set %s to your campus gateway base URL (ending in /v1), or pass base_url=.",
        cfg$base_url_env), call. = FALSE)
      if (is.null(model) || !nzchar(model)) stop(
        "The institutional gateway requires model = a model name your gateway serves (ask IT).",
        call. = FALSE)
      chat_openai_compatible(
        base_url = url, name = "Institutional gateway",
        model = model, system_prompt = system_prompt, credentials = function() key, ...
      )
    },

    # Azure OpenAI: endpoint from env, model = deployment name, key-authenticated.
    azure = {
      endpoint <- Sys.getenv(cfg$endpoint_env)
      if (!nzchar(endpoint)) stop(sprintf(
        "No Azure endpoint. Set %s to your Azure OpenAI resource endpoint (https://<resource>.openai.azure.com).",
        cfg$endpoint_env), call. = FALSE)
      if (is.null(model) || !nzchar(model)) stop(
        "Azure OpenAI requires model = your deployment name (e.g. \"gpt-4o\").", call. = FALSE)
      chat_azure_openai(
        endpoint = endpoint, model = model, api_version = cfg$api_version,
        system_prompt = system_prompt, credentials = function() key, ...
      )
    },

    # Vertex AI: project + location; auth via gcloud Application Default Credentials.
    vertex = {
      project  <- .first_env(cfg$project_env)
      location <- .first_env(cfg$location_env)
      if (!nzchar(project) || !nzchar(location)) stop(paste0(
        "Vertex AI needs a project and location. Set VERTEX_PROJECT/GOOGLE_CLOUD_PROJECT and ",
        "VERTEX_LOCATION/GOOGLE_CLOUD_LOCATION, then authenticate once with:\n",
        "  gcloud auth application-default login"), call. = FALSE)
      chat_google_vertex(
        location = location, project_id = project,
        model = model, system_prompt = system_prompt, ...
      )
    },

    # AWS Bedrock: AWS credential chain (optional AWS_PROFILE); model = Bedrock id.
    bedrock = {
      if (is.null(model) || !nzchar(model)) stop(
        "AWS Bedrock requires model = a Bedrock model id (e.g. \"anthropic.claude-3-5-sonnet-20241022-v2:0\").",
        call. = FALSE)
      profile <- Sys.getenv(cfg$profile_env)
      cargs <- list(model = model, system_prompt = system_prompt, ...)
      if (nzchar(profile)) cargs$profile <- profile
      do.call(chat_aws_bedrock, cargs)
    },

    # Local Ollama: no key; base_url defaults to localhost.
    ollama = {
      url <- base_url %||% Sys.getenv(cfg$base_url_env)
      cargs <- list(model = model, system_prompt = system_prompt, ...)
      if (nzchar(url)) cargs$base_url <- url
      do.call(chat_ollama, cargs)
    }
  )
}

# Convenience: set a provider's API key in this session without remembering the var.
#   set_api_key("claude", "sk-ant-...")
# (No-op for providers that do not use a simple key: vertex, bedrock, ollama.)
set_api_key <- function(provider = names(.PROVIDERS), key) {
  provider <- match.arg(provider)
  cfg <- .PROVIDERS[[provider]]
  if (is.null(cfg$keys)) {
    message(sprintf("%s does not use a simple API key (auth: %s). Nothing set.",
                    cfg$label, cfg$auth))
    return(invisible(NULL))
  }
  var <- cfg$keys[1]
  args <- list(key); names(args) <- var
  do.call(Sys.setenv, args)
  invisible(var)
}

# List supported providers, their default models, auth style, and what credential
# each needs.
list_providers <- function() {
  cred_hint <- function(p) {
    switch(p$auth,
      key        = paste(p$keys, collapse = " | "),
      google_adc = "gcloud ADC + VERTEX_PROJECT / VERTEX_LOCATION",
      aws        = "AWS credential chain (AWS_PROFILE / AWS_* keys)",
      none       = "none (local server)",
      p$auth
    )
  }
  data.frame(
    provider      = names(.PROVIDERS),
    label         = vapply(.PROVIDERS, `[[`, "", "label"),
    default_model = vapply(.PROVIDERS, function(p) p$default %||% "(set model)", ""),
    auth          = vapply(.PROVIDERS, `[[`, "", "auth"),
    credential    = vapply(.PROVIDERS, cred_hint, ""),
    row.names = NULL
  )
}
