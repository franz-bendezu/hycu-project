import { ChangeEvent, DragEvent, useRef, useState } from "react";

import { StatusPill } from "./StatusPill";

export type QueuedImage = {
  id: string;
  name: string;
  size: number;
  file: File;
  previewUrl: string;
  status: "ready" | "analyzing" | "complete" | "failed";
  assetId?: string;
  jobId?: string;
  message?: string;
};

type ImageBatchUploaderProps = {
  queuedImages: QueuedImage[];
  onAddFiles: (files: File[]) => void;
  onRemoveFile: (id: string) => Promise<void>;
  onClear: () => void;
  onAnalyze: () => Promise<void>;
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function ImageBatchUploader({
  queuedImages,
  onAddFiles,
  onRemoveFile,
  onClear,
  onAnalyze,
}: ImageBatchUploaderProps): React.JSX.Element {
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  function handleFiles(files: FileList | null): void {
    if (!files) {
      return;
    }
    onAddFiles(Array.from(files));
  }

  function onFileInputChange(event: ChangeEvent<HTMLInputElement>): void {
    handleFiles(event.target.files);
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    setDragActive(false);
    handleFiles(event.dataTransfer.files);
  }

  function onDragOver(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    setDragActive(true);
  }

  function onDragLeave(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    setDragActive(false);
  }

  return (
    <div className="upload-shell">
      <div
        className={`dropzone ${dragActive ? "active" : ""}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        role="button"
        tabIndex={0}
        onClick={() => fileInputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            fileInputRef.current?.click();
          }
        }}
      >
        <p className="dropzone-title">Drag and drop images here</p>
        <p className="dropzone-subtitle">or click to select multiple JPG, PNG, WEBP files</p>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={onFileInputChange}
        />
      </div>

      <div className="upload-actions">
        <button type="button" onClick={() => fileInputRef.current?.click()}>
          Add images
        </button>
        <button
          type="button"
          className="secondary"
          onClick={onAnalyze}
          disabled={queuedImages.length === 0}
        >
          Analyze queued images
        </button>
        <button
          type="button"
          className="secondary"
          onClick={onClear}
          disabled={queuedImages.length === 0}
        >
          Clear queue
        </button>
      </div>

      {queuedImages.length > 0 ? (
        <ul className="upload-list">
          {queuedImages.map((item) => (
            <li key={item.id} className="upload-item">
              <img src={item.previewUrl} alt={item.name} className="upload-thumb" />
              <div>
                <strong>{item.name}</strong>
                <p>{formatBytes(item.size)}</p>
                <StatusPill
                  label={item.status}
                  tone={
                    item.status === "complete"
                      ? "ok"
                      : item.status === "failed"
                        ? "error"
                        : item.status === "analyzing"
                          ? "warning"
                          : "neutral"
                  }
                />
                {item.message ? <p className="upload-message">{item.message}</p> : null}
              </div>
              <button type="button" className="secondary" onClick={() => void onRemoveFile(item.id)}>
                Remove
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
