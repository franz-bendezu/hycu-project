import { Link } from "react-router-dom";

export function NotFoundView(): React.JSX.Element {
  return (
    <section className="panel">
      <h2>Page not found</h2>
      <p>This route does not exist in the modular workspace.</p>
      <Link to="/" className="button secondary">
        Back to workspace
      </Link>
    </section>
  );
}
