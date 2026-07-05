use crate::api::ApiClient;
use crate::output::{self, OutputFormat};
use crate::ui;
use anyhow::Result;

pub fn list(
    client: &ApiClient,
    agent_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching operations..."))
    } else {
        None
    };

    let response = client.list_operations(agent_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(ops_response) => {
            if output_format == OutputFormat::Pretty {
                if ops_response.operations.is_empty() {
                    ui::print_info("No operations found");
                } else {
                    ui::print_info(&format!(
                        "Found {} operation(s)",
                        ops_response.operations.len()
                    ));
                    for op in &ops_response.operations {
                        println!("\n  Operation ID: {}", op.id);
                        println!("    Type: {}", op.task_type);
                        println!("    Status: {}", op.status);
                        println!("    Items: {}", op.items_count);
                        if let Some(doc_id) = &op.document_id {
                            println!("    Document ID: {}", doc_id);
                        }
                    }
                }
            } else {
                output::print_output(&ops_response, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Get the status of a specific operation
pub fn get(
    client: &ApiClient,
    agent_id: &str,
    operation_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching operation status..."))
    } else {
        None
    };

    let response = client.get_operation(agent_id, operation_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_section_header(&format!("Operation: {}", operation_id));

                use hms_client::types::Status;
                let status_str = match &result.status {
                    Status::Completed => ui::gradient_start("completed"),
                    Status::Pending => ui::gradient_mid("pending"),
                    Status::Processing => ui::gradient_mid("processing"),
                    Status::Failed => ui::gradient_end("failed"),
                    Status::Cancelled => ui::dim("cancelled"),
                    Status::NotFound => ui::gradient_end("not_found"),
                };

                println!("  {} {}", ui::dim("Status:"), status_str);

                if let Some(error) = &result.error_message {
                    println!("  {} {}", ui::dim("Error:"), ui::gradient_end(error));
                }

                println!();
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

pub fn cancel(
    client: &ApiClient,
    agent_id: &str,
    operation_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Cancelling operation..."))
    } else {
        None
    };

    let response = client.cancel_operation(agent_id, operation_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                if result.success {
                    ui::print_success("Operation cancelled successfully");
                } else {
                    ui::print_error("Failed to cancel operation");
                }
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Retry a failed async operation
pub fn retry(
    client: &ApiClient,
    agent_id: &str,
    operation_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Retrying operation..."))
    } else {
        None
    };

    let response = client.retry_operation(agent_id, operation_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    let result = response?;
    if output_format == OutputFormat::Pretty {
        ui::print_success(&format!("Operation '{}' retried", operation_id));
        let json = serde_json::to_value(&result)?;
        println!(
            "  {}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::print_output(&result, output_format)?;
    }
    Ok(())
}
