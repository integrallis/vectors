import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.integrallis.models.api.ModelPrompt;
import java.nio.file.Files;
import java.nio.file.Path;

/** Re-renders the model-check base-arm prompts and verifies each against its recorded SHA-256. */
public class DumpPrompts {
  public static void main(String[] args) throws Exception {
    ObjectMapper json = new ObjectMapper();
    JsonNode window = json.readTree(Path.of(args[0]).toFile());
    JsonNode recorded = json.readTree(Path.of(args[1]).toFile());
    JsonNode suite = null;
    for (JsonNode s : window.path("suites")) if (s.path("name").asText().equals("squad-v2-dev")) suite = s;
    ArrayNode out = json.createArrayNode();
    int index = 0, mismatches = 0;
    for (JsonNode rec : recorded.path("cases")) {
      JsonNode item = null;
      for (JsonNode c : suite.path("cases")) if (c.path("id").asText().equals(rec.path("id").asText())) item = c;
      ModelPrompt prompt = ModelCheck.prompt(item, ModelCheck.BASE_INSTRUCTION);
      StringBuilder text = new StringBuilder();
      for (ModelPrompt.Segment segment : prompt.segments()) text.append(segment.text());
      String sha = ModelCheck.sha256(text.toString());
      boolean ok = sha.equals(rec.path("promptTextSha256").asText());
      if (!ok) mismatches++;
      ObjectNode row = json.createObjectNode();
      row.put("id", rec.path("id").asText()); row.put("promptText", text.toString()); row.put("sha256", sha); row.put("shaMatches", ok);
      row.put("promptTokens", rec.path("promptTokens").asInt());
      out.add(row); index++;
    }
    json.writerWithDefaultPrettyPrinter().writeValue(Path.of(args[2]).toFile(), out);
    System.out.println("prompts=" + index + " shaMismatches=" + mismatches);
    if (mismatches > 0) System.exit(3);
  }
}
