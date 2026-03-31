import { NextRequest, NextResponse } from 'next/server';
import { checkAuth, getBackendHeaders, BACKEND_URL } from '../../auth';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ exp_dir: string }> }
) {
  // Apply rate limiting
  const authError = checkAuth(request);
  if (authError) return authError;

  const { exp_dir } = await params;
  try {
    const response = await fetch(`${BACKEND_URL}/info/${exp_dir}`, {
      cache: 'no-store',
      headers: getBackendHeaders(),
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: `Backend error: ${response.status}` },
        { status: response.status }
      );
    }

    const data = await response.json();

    // Transform backend response to match frontend expectations
    // Backend returns: name, path, checkpoint_exists, sample_output, model_info, git_log
    // Frontend expects: exp_dir, checkpoint_path, has_checkpoint, etc.
    const transformedData = {
      exp_dir: data.name || exp_dir,
      checkpoint_path: data.checkpoint_exists ? `${BACKEND_URL}/checkpoints/${data.name}/checkpoint.pt` : null,
      has_checkpoint: data.checkpoint_exists,
      has_program: true,
      has_results: true,
      has_git_log: !!data.git_log,
      has_global_results: true,
      has_metadata: !!data.model_info,
      git_log: data.git_log ? [{
        commit_sha: '',
        commit_short: exp_dir.split('-')[1] || '',
        author: '',
        date: '',
        message: data.git_log
      }] : [],
      results_data: null,
      metadata: data.model_info ? parseModelInfo(data.model_info) : null,
      sample_output: data.sample_output
    };

    return NextResponse.json(transformedData);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to connect to backend' },
      { status: 502 }
    );
  }
}

// Parse model_info.txt content into a structured object
function parseModelInfo(text: string): Record<string, any> {
  const result: Record<string, any> = {};
  text.split('\n').forEach(line => {
    const [key, ...valueParts] = line.split(':');
    if (key && valueParts.length > 0) {
      result[key.trim()] = valueParts.join(':').trim();
    }
  });
  return result;
}