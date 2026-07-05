{{/*
Expand the name of the chart.
*/}}
{{- define "hms.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "hms.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "hms.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "hms.labels" -}}
helm.sh/chart: {{ include "hms.chart" . }}
{{ include "hms.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "hms.selectorLabels" -}}
app.kubernetes.io/name: {{ include "hms.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
API labels
*/}}
{{- define "hms.api.labels" -}}
{{ include "hms.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
API selector labels
*/}}
{{- define "hms.api.selectorLabels" -}}
{{ include "hms.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
Control plane labels
*/}}
{{- define "hms.controlPlane.labels" -}}
{{ include "hms.labels" . }}
app.kubernetes.io/component: control-plane
{{- end }}

{{/*
Control plane selector labels
*/}}
{{- define "hms.controlPlane.selectorLabels" -}}
{{ include "hms.selectorLabels" . }}
app.kubernetes.io/component: control-plane
{{- end }}

{{/*
Worker labels
*/}}
{{- define "hms.worker.labels" -}}
{{ include "hms.labels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Worker selector labels
*/}}
{{- define "hms.worker.selectorLabels" -}}
{{ include "hms.selectorLabels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "hms.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "hms.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Generate database URL
*/}}
{{- define "hms.databaseUrl" -}}
{{- if .Values.databaseUrl }}
{{- .Values.databaseUrl }}
{{- else if .Values.postgresql.enabled }}
{{- printf "postgresql://%s:%s@%s-postgresql:%d/%s" .Values.postgresql.auth.username .Values.postgresql.auth.password (include "hms.fullname" .) (.Values.postgresql.service.port | int) .Values.postgresql.auth.database }}
{{- else }}
{{- printf "postgresql://%s:$(POSTGRES_PASSWORD)@%s:%d/%s" .Values.postgresql.external.username .Values.postgresql.external.host (.Values.postgresql.external.port | int) .Values.postgresql.external.database }}
{{- end }}
{{- end }}

{{/*
API URL for control plane
*/}}
{{- define "hms.apiUrl" -}}
{{- printf "http://%s-api:%d" (include "hms.fullname" .) (.Values.api.service.port | int) }}
{{- end }}

{{/*
TEI reranker labels
*/}}
{{- define "hms.tei.reranker.labels" -}}
{{ include "hms.labels" . }}
app.kubernetes.io/component: tei-reranker
{{- end }}

{{/*
TEI reranker selector labels
*/}}
{{- define "hms.tei.reranker.selectorLabels" -}}
{{ include "hms.selectorLabels" . }}
app.kubernetes.io/component: tei-reranker
{{- end }}

{{/*
TEI embedding labels
*/}}
{{- define "hms.tei.embedding.labels" -}}
{{ include "hms.labels" . }}
app.kubernetes.io/component: tei-embedding
{{- end }}

{{/*
TEI embedding selector labels
*/}}
{{- define "hms.tei.embedding.selectorLabels" -}}
{{ include "hms.selectorLabels" . }}
app.kubernetes.io/component: tei-embedding
{{- end }}

{{/*
Get the name of the secret to use
*/}}
{{- define "hms.secretName" -}}
{{- if .Values.existingSecret }}
{{- .Values.existingSecret }}
{{- else }}
{{- printf "%s-secret" (include "hms.fullname" .) }}
{{- end }}
{{- end }}
